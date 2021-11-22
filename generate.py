# Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree.
import json
import logging
from argparse import ArgumentParser
from pathlib import Path
from pprint import pformat
from typing import Dict, Iterable, Optional

import torch
import torch.nn.functional as F
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel, PretrainedConfig, GPT2LMHeadModel, GPT2Tokenizer, top_k_top_p_filtering
from transformers.file_utils import ModelOutput

from .conv_data import ConversationalDataset, Collator, SpecialTokens, SPECIAL_TOKENS, add_special_tokens
from .log_utils import is_wandb_available, authorize_wandb


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop


def generate_no_beam_search(
    model: PreTrainedModel,
    input_ids: torch.LongTensor,
    decoder_input_ids: Optional[torch.LongTensor] = None,
    max_length: Optional[int] = None,
    min_length: Optional[int] = None,
    do_sample: Optional[bool] = None,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
    bad_words_ids: Optional[Iterable[int]] = None,
    bos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    no_repeat_ngram_size: Optional[int] = None,
    num_return_sequences: Optional[int] = None,
    attention_mask: Optional[torch.LongTensor] = None,
    decoder_start_token_id: Optional[int] = None,
    use_cache: Optional[bool] = None,
    token_type_ids: Optional[torch.LongTensor] = None,
    **model_kwargs,
) -> Dict[str, torch.LongTensor]:
    """Generate sequences for each example without beam search (num_beams == 1).
    All returned sequence are generated independantly.
    """
    config: PretrainedConfig = getattr(model, "module", model).config

    max_length = max_length if max_length is not None else config.max_length
    min_length = min_length if min_length is not None else config.min_length
    do_sample = do_sample if do_sample is not None else config.do_sample
    use_cache = use_cache if use_cache is not None else config.use_cache
    temperature = temperature if temperature is not None else config.temperature
    top_k = top_k if top_k is not None else config.top_k
    top_p = top_p if top_p is not None else config.top_p
    repetition_penalty = repetition_penalty if repetition_penalty is not None else config.repetition_penalty
    bos_token_id = bos_token_id if bos_token_id is not None else config.bos_token_id
    pad_token_id = pad_token_id if pad_token_id is not None else config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else config.eos_token_id
    no_repeat_ngram_size = no_repeat_ngram_size if no_repeat_ngram_size is not None else config.no_repeat_ngram_size
    bad_words_ids = bad_words_ids if bad_words_ids is not None else config.bad_words_ids
    num_return_sequences = num_return_sequences if num_return_sequences is not None else config.num_return_sequences
    decoder_start_token_id = (
        decoder_start_token_id if decoder_start_token_id is not None else config.decoder_start_token_id
    )

    batch_size = input_ids.shape[0]

    if (attention_mask is None) and (pad_token_id is not None) and (pad_token_id in input_ids):
        attention_mask = input_ids.ne(pad_token_id).long()
    elif attention_mask is None:
        attention_mask = input_ids.new_ones(input_ids.shape)

    if pad_token_id is None and eos_token_id is not None:
        logger.warning("Setting `pad_token_id` to {} (first `eos_token_id`) to generate sequence".format(eos_token_id))
        pad_token_id = eos_token_id

    # set effective batch size and effective batch multiplier according to do_sample

    effective_batch_size = batch_size * num_return_sequences
    effective_batch_mult = num_return_sequences

    if config.is_encoder_decoder:
        if decoder_start_token_id is None:
            # see if BOS token can be used for decoder_start_token_id
            if bos_token_id is not None:
                decoder_start_token_id = bos_token_id
            elif (
                hasattr(config, "decoder")
                and hasattr(config.decoder, "bos_token_id")
                and config.decoder.bos_token_id is not None
            ):
                decoder_start_token_id = config.decoder.bos_token_id
            else:
                raise ValueError(
                    "decoder_start_token_id or bos_token_id has to be defined for encoder-decoder generation"
                )

        assert hasattr(model, "get_encoder"), "{} should have a 'get_encoder' function defined".format(model)
        assert callable(model.get_encoder), "{} should be a method".format(model.get_encoder)

        # get encoder and store encoder outputs
        encoder = model.get_encoder()
        encoder_outputs: ModelOutput = encoder(input_ids, attention_mask=attention_mask, return_dict=True)

    # Expand input ids if num_return_sequences > 1
    if num_return_sequences > 1:
        input_ids_len = input_ids.shape[-1]
        input_ids = input_ids.unsqueeze(1).expand(batch_size, effective_batch_mult, input_ids_len)
        attention_mask = attention_mask.unsqueeze(1).expand(batch_size, effective_batch_mult, input_ids_len)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.unsqueeze(1).expand(-1, effective_batch_mult, input_ids_len)
            token_type_ids = token_type_ids.contiguous().view(effective_batch_size, input_ids_len)

        input_ids = input_ids.contiguous().view(
            effective_batch_size, input_ids_len
        )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)
        attention_mask = attention_mask.contiguous().view(
            effective_batch_size, input_ids_len
        )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)

    if config.is_encoder_decoder:
        device = next(model.parameters()).device
        if decoder_input_ids is not None:
            # give initial decoder input ids
            input_ids = decoder_input_ids.repeat(effective_batch_size, 1).to(device)
        else:
            # create empty decoder input_ids
            input_ids = torch.full(
                (effective_batch_size, 1),
                decoder_start_token_id,
                dtype=torch.long,
                device=device,
            )

        assert (
            batch_size == encoder_outputs.last_hidden_state.shape[0]
        ), f"expected encoder_outputs.last_hidden_state to have 1st dimension bs={batch_size}, got {encoder_outputs.last_hidden_state.shape[0]} "

        # expand batch_idx to assign correct encoder output for expanded input_ids (due to num_beams > 1 and num_return_sequences > 1)
        expanded_batch_idxs = (
            torch.arange(batch_size).view(-1, 1).repeat(1, effective_batch_mult).view(-1).to(input_ids.device)
        )

        # expand encoder_outputs
        encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(0, expanded_batch_idxs)

        # save encoder_outputs in `model_kwargs`
        model_kwargs["encoder_outputs"] = encoder_outputs

    input_lengths = (input_ids != pad_token_id).int().sum(-1) - 1

    # length of generated sentences / unfinished sentences
    unfinished_sents = input_ids.new(effective_batch_size).fill_(1)
    sent_lengths = input_ids.new(effective_batch_size).fill_(max_length)

    generated_ids = input_ids[torch.arange(effective_batch_size), input_lengths].unsqueeze(-1)

    if token_type_ids is not None:
        generated_token_types = token_type_ids[torch.arange(effective_batch_size), input_lengths].unsqueeze(-1)

    past = None
    for cur_len in range(max_length):
        model_inputs = model.prepare_inputs_for_generation(
            input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache, **model_kwargs
        )

        if token_type_ids is not None:
            if past:
                model_inputs["token_type_ids"] = token_type_ids[:, -1].unsqueeze(-1)
            else:
                model_inputs["token_type_ids"] = token_type_ids

        outputs = model(**model_inputs, return_dict=True)
        if cur_len == 0:
            next_token_logits = outputs.logits[torch.arange(effective_batch_size), input_lengths, :]
        else:
            next_token_logits = outputs.logits[:, -1, :]

        scores = model.postprocess_next_token_scores(
            scores=next_token_logits,
            input_ids=input_ids,
            no_repeat_ngram_size=no_repeat_ngram_size,
            bad_words_ids=bad_words_ids,
            cur_len=cur_len,
            min_length=min_length,
            max_length=max_length,
            eos_token_id=eos_token_id,
            repetition_penalty=repetition_penalty,
            batch_size=batch_size,
            num_beams=1,
        )

        # if model has past, then set the past variable to speed up decoding
        if "past_key_values" in outputs:
            past = outputs.past_key_values
        elif "mems" in outputs:
            past = outputs.mems

        if do_sample:
            # Temperature (higher temperature => more likely to sample low probability tokens)
            if temperature != 1.0:
                scores = scores / temperature
            # Top-p/top-k filtering
            next_token_logscores = top_k_top_p_filtering(scores, top_k=top_k, top_p=top_p)
            # Sample
            probs = F.softmax(next_token_logscores, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            # Greedy decoding
            if cur_len == 0:
                _, next_token = torch.topk(
                    next_token_logits.view(batch_size, effective_batch_mult, -1)[:, 0, :],
                    k=effective_batch_mult,
                    dim=-1,
                )
                next_token = next_token.reshape(effective_batch_size, -1).squeeze(-1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1)

        # update generations and finished sentences
        if eos_token_id is not None:
            # pad finished sentences if eos_token_id exist
            tokens_to_add = next_token * unfinished_sents + (pad_token_id) * (1 - unfinished_sents)
        else:
            tokens_to_add = next_token

        # add token and increase length by one
        if token_type_ids is not None:
            next_token_types = torch.gather(token_type_ids, dim=1, index=input_lengths.unsqueeze(-1)).squeeze(
                -1
            ) * unfinished_sents + pad_token_id * (1 - unfinished_sents)
        next_len = cur_len + 1
        input_lengths = input_lengths + 1
        input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)

        generated_ids = torch.cat([generated_ids, tokens_to_add.unsqueeze(-1)], dim=-1)

        if token_type_ids is not None:
            token_type_ids = torch.cat([token_type_ids, next_token_types.unsqueeze(-1)], dim=-1)
            generated_token_types = torch.cat([generated_token_types, next_token_types.unsqueeze(-1)], dim=-1)

        if eos_token_id is not None:
            eos_in_sents = tokens_to_add == eos_token_id
            # if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
            is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
            sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, next_len)
            # unfinished_sents is set to zero if eos in sentence
            unfinished_sents.mul_((~eos_in_sents).long())

        # stop when there is a </s> in each sentence, or if we exceed the maximul length
        if unfinished_sents.max() == 0:
            break

        # extend attention_mask for new generated input if only decoder
        if model.config.is_encoder_decoder is False:
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)

    output = dict(input_ids=input_ids, generated_ids=generated_ids, attention_mask=attention_mask)
    if token_type_ids is not None:
        output["token_type_ids"] = token_type_ids
        output["generated_token_types"] = generated_token_types

    return output


def get_data_loader(args, tokenizer) -> DataLoader:
    """ Prepare the dataset for training and evaluation """
    test_dataset = ConversationalDataset(
        args.dataset_path,
        tokenizer,
        max_history=args.max_history,
        exclude_kb=args.exclude_kb,
        is_generation=True,
    )
    logger.info("Build inputs")
    collator = Collator(tokenizer.pad_token_id)

    logger.info("Build test dataloader")

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
    )

    logger.info(f"Test dataset size: {len(test_dataset)}")
    return test_loader


def concat_padded_tensors(t1: torch.Tensor, t2: torch.Tensor, pad_token: int = 0) -> torch.Tensor:
    assert len(t1.shape) == len(t2.shape)
    assert all(d1 == d2 for d1, d2 in zip(t1.shape[:-1], t2.shape[:-1]))
    concat_dim = t1.shape[-1] + t2.shape[-1]
    v1 = t1.reshape(-1, t1.shape[-1])
    v2 = t2.reshape(-1, t2.shape[-1])
    out = v1.new_full((v1.shape[0], concat_dim), pad_token)
    out[:, : v1.shape[1]] = v1

    for i in range(v2.shape[0]):
        t1_row = (v1[i] != pad_token).nonzero(as_tuple=False).max() + 1
        t2_nonpads = (v2[i] != pad_token).nonzero(as_tuple=False)
        t2_row = (t2_nonpads.max() + 1) if t2_nonpads.nelement() > 0 else 0
        if t2_row > 0:
            out[i, t1_row : t1_row + t2_row] = v2[i, :t2_row]

    return out.reshape(*t1.shape[:-1], concat_dim)


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the JsonL dataset.")
    parser.add_argument("--model_checkpoint", type=str, required=True, help="Path to a trained model")
    parser.add_argument("--output", type=str, default="", help="Path of the output directory to save the responses")
    parser.add_argument(
        "--max_history",
        type=int,
        default=2,
        help="Number of previous exchanges to keep in history",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for validation")
    parser.add_argument("--max_length", type=int, default=100)
    parser.add_argument("--min_length", type=int, default=2)
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="value used to module the next token probabilities"
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        default=False,
        help="Whether or not to use sampling ; use greedy decoding otherwise.",
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2"
    )
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument(
        "--exclude_kb",
        action="store_true",
        default=False,
        help="Whether to exclude knowledge from input sequences",
    )
    parser.add_argument("--namestr", type=str, default="exp1", help="additional info to describe experiments")
    parser.add_argument("--wandb", action="store_true", default=False, help="Use wandb for logging")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda or cpu)",
    )

    args = parser.parse_args()

    # logging is set to INFO (resp. WARN) for main (resp. auxiliary) process. logger.info => log main process only, logger.warning => log all processes
    logging.basicConfig(level=logging.INFO)
    logger.info("Arguments: %s", pformat(args))
    logger.info("Prepare tokenizer and pretrained model.")

    # WANDB logger
    if args.wandb and is_wandb_available():
        import wandb

        wandb.init(
            project="DialogueKB",
            name="Finetune-{}".format(args.namestr),
            anonymous="never" if authorize_wandb() else "allow",
        )

    tokenizer = GPT2Tokenizer.from_pretrained(args.model_checkpoint)
    model = GPT2LMHeadModel.from_pretrained(args.model_checkpoint)
    model.to(args.device)

    add_special_tokens(tokenizer, model)
    if args.max_length < 0 and model.config.max_position_embeddings > 0:
        args.max_length = model.config.max_position_embeddings
    elif 0 < model.config.max_position_embeddings < args.max_length:
        args.max_length = model.config.max_position_embeddings  # No generation bigger than model size
    elif args.max_length < 0:
        args.max_length = MAX_LENGTH  # avoid infinite loop

    if torch.cuda.device_count() > 1:
        model = DataParallel(model)

    test_loader = get_data_loader(args, tokenizer)

    # Evaluation function and evaluator (evaluator output is the input of the metrics)
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(args.model_checkpoint) / "gen_output"
        output_dir.mkdir(exist_ok=True)

    out_file = (
        output_dir / f"{Path(args.dataset_path).with_suffix('').name}_generated_"
        f"len{args.max_length}_temp{args.temperature}_p{args.top_p}_k{args.top_k}.jsonl"
    )
    logger.info(f"Results will be saved in `{out_file}`")

    with out_file.open("w", encoding="utf-8") as writer:
        special_tokens = SpecialTokens(*tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS))
        for batch in tqdm(test_loader, total=len(test_loader)):
            model.eval()
            batch = {k: t.to(args.device) for k, t in batch.items()}
            input_ids = batch["input_ids"]
            token_type_ids = batch["token_type_ids"]
            batch_size = input_ids.shape[0]
            with torch.no_grad():
                # responses = (batch_size * num_return_sequences, sequence_length)
                responses = generate_no_beam_search(
                    model,
                    input_ids,
                    max_length=args.max_length,
                    min_length=args.min_length,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    bos_token_id=tokenizer.bos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    num_return_sequences=args.num_return_sequences,
                    use_cache=True,
                    token_type_ids=token_type_ids,
                )

            response_ids = responses["generated_ids"]
            response_token_types = responses["generated_token_types"]

            with torch.no_grad():
                # responses = (batch_size * num_return_sequences, sequence_length)
                triples = generate_no_beam_search(
                    model,
                    input_ids2,
                    max_length=args.max_length,
                    min_length=args.min_length,
                    do_sample=True,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    bos_token_id=special_tokens.triple,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    num_return_sequences=args.num_return_sequences,
                    use_cache=True,
                    token_type_ids=token_types2,
                )

            # generated_ids = (batch_size, num_return_sequences, sequence_length)
            generated_ids = responses["generated_ids"].reshape(batch_size, args.num_return_sequences, -1).cpu().numpy()
            triple_ids = triples["generated_ids"].reshape(batch_size, args.num_return_sequences, -1).cpu().numpy()

            with open(args.dataset_path, "r", encoding="utf-8") as f:
                for b in range(batch_size):
                    data = json.loads(f.readline())
                    data["history"] = data["history"][-args.max_history :]
                    if not data["knowledge_base"]:
                        continue
                    if not args.exclude_kb and data["knowledge_base"]:
                        kb = data.pop("knowledge_base")
                        data["knowledge_base"] = [triple for triple in kb["paths"]]

                    data["generated_response"] = [
                        tokenizer.decode(
                            out,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        )
                        for out in generated_ids[b]
                    ]

                    data["cycle_triple"] = [
                        tokenizer.decode(
                            triple,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        )
                        for triple in triple_ids[b]
                    ]

                    writer.write(json.dumps(data) + "\n")


if __name__ == "__main__":
    main()
