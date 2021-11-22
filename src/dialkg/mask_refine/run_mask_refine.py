import json
import logging
import os
import shutil
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

import pytorch_lightning as pl
import spacy
import torch
import torch.nn.functional as F
from torch_scatter import scatter_sum
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForTokenClassification,
    AutoConfig,
    AutoTokenizer,
    T5TokenizerFast,
    T5Config,
    T5ForConditionalGeneration,
)

from .data import (
    _normalize,
    HallucOnlyCollator,
    NER,
    MaskRefineBatch,
    GenPipelineBatch,
    MaskRefineCollator,
    MaskRerankCollator,
    GenPipelineCollator,
    MixedCollator,
    MaskRefineDataset,
    MaskRefineInferenceDataset,
    MaskRerankInferenceDataset,
    GenPipelineDataset,
    MixedDataset,
    MultiTaskBatch,
)

from .modeling_mask_refine import MaskRefineModel
from ..conv_data import (
    ConversationalDataset,
    Collator as LMCollator,
    SpecialTokens,
    SPECIAL_TOKENS,
    add_special_tokens,
)
from ..generate import generate_no_beam_search
from ..kge import KnowledgeGraphEmbedding
from ..lightning_base import BaseTransformer, generic_train, add_generic_args
from ..log_utils import is_wandb_available, authorize_wandb


logger = logging.getLogger("run_mask_refine")


class NERLabel(Enum):
    O = ("O", 0)
    B = ("B-Halluc", 1)
    I = ("I-Halluc", 2)

    @classmethod
    def of(cls, label_id: int) -> "NERLabel":
        for lbl in NERLabel:
            if lbl.value[1] == label_id:
                return lbl
        raise ValueError(f"label_id not found: {label_id}")

    @classmethod
    def map(cls) -> Dict[int, str]:
        return {l.value[1]: l.value[0] for l in NERLabel}


class DialogueMaskRefineTransformer(BaseTransformer):
    mode = "language-modeling"
    kge: Optional[KnowledgeGraphEmbedding] = None

    def __init__(self, hparams: Namespace):
        """Initialize a model, tokenizer and config."""
        self.kge = KnowledgeGraphEmbedding(hparams.kge, hparams.train_dataset_path, hparams.eval_dataset_path)
        self.ner = NER(hparams.output_dir, graph_file=hparams.graph_file)

        if os.path.exists(hparams.model_name_or_path):
            best_model_dir = os.path.join(hparams.model_name_or_path, "best_model")
            if os.path.exists(best_model_dir):
                hparams.model_name_or_path = os.path.join(best_model_dir, "lm")
                if os.path.exists(os.path.join(best_model_dir, "mlm")):
                    hparams.mlm_model = os.path.join(best_model_dir, "mlm")

        super().__init__(hparams, self.mode, return_dict=True, output_hidden_states=True)

        self.nlp = spacy.load("en_core_web_sm", disable=("ner", "parser", "tagger", "lemmatizer"))

        add_special_tokens(self.tokenizer, self.model)
        self.special_tokens = SpecialTokens(*self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS))
        self.config.pad_token_id = self.tokenizer.pad_token_id

        if not self.hparams.bypass_mlm:
            self.mlm_config = AutoConfig.from_pretrained(
                self.hparams.mlm_model, cache_dir=self.hparams.cache_dir, return_dict=True, output_hidden_states=True
            )
            self.mlm_tokenizer = AutoTokenizer.from_pretrained(
                self.hparams.mlm_model, cache_dir=self.hparams.cache_dir
            )
            self.mlm_model = AutoModelForMaskedLM.from_pretrained(
                self.hparams.mlm_model, config=self.mlm_config, cache_dir=self.hparams.cache_dir
            )
        else:
            self.mlm_tokenizer = None
            self.mlm_model = None

        self.nce_dataset = None
        self.nce_valid_dataset = None
        if self.hparams.train_dataset_path is not None and self.hparams.do_train:
            self.nce_dataset = MaskRefineDataset(
                self.hparams.train_dataset_path,
                self.tokenizer,
                self.mlm_tokenizer,
                self.ner,
                self.kge,
                self.hparams.max_adjacents,
                self.hparams.max_history,
                self.hparams.max_seq_length,
                self.hparams.exclude_kb,
                self.hparams.include_render,
                do_lm=not self.hparams.multitask,
            )

            self.nce_valid_dataset = MaskRefineDataset(
                self.hparams.eval_dataset_path,
                self.tokenizer,
                self.mlm_tokenizer,
                self.ner,
                self.kge,
                self.hparams.max_adjacents,
                self.hparams.max_history,
                self.hparams.max_seq_length,
                self.hparams.exclude_kb,
                self.hparams.include_render,
                do_lm=True,
            )

        self.mask_refine_model = MaskRefineModel.new_or_from_pretrained(
            self.hparams.model_name_or_path,
            self.model,
            self.mlm_model,
            self.kge.node_embds,
            self.kge.rel_embds,
            inbatch_negatives=self.hparams.inbatch_negatives,
            stabilize=not self.hparams.no_stabilize,
        )

        self.halluc_config = None
        self.halluc_tokenizer = None
        self.halluc_classifier = None
        if self.hparams.do_generate:
            if not self.hparams.rerank_only and not self.hparams.skip_halluc:
                assert self.hparams.halluc_classifier is not None

                self.halluc_config = AutoConfig.from_pretrained(
                    self.hparams.halluc_classifier,
                    cache_dir=self.hparams.cache_dir,
                    return_dict=True,
                    output_hidden_states=True,
                )
                self.halluc_tokenizer = AutoTokenizer.from_pretrained(
                    self.hparams.halluc_classifier, cache_dir=self.hparams.cache_dir
                )
                self.halluc_classifier = AutoModelForTokenClassification.from_pretrained(
                    self.hparams.halluc_classifier, config=self.halluc_config, cache_dir=self.hparams.cache_dir
                )

            if self.hparams.rerank_name_or_path is not None:
                self.rerank_config = AutoConfig.from_pretrained(
                    self.hparams.rerank_name_or_path, cache_dir=self.hparams.cache_dir
                )
                self.rerank_tokenizer = AutoTokenizer.from_pretrained(
                    self.hparams.rerank_name_or_path, cache_dir=self.hparams.cache_dir
                )

                if self.rerank_config.model_type == "t5":
                    self.rerank_model = T5ForConditionalGeneration.from_pretrained(
                        self.hparams.rerank_name_or_path, cache_dir=self.hparams.cache_dir, return_dict=True
                    )
                else:
                    self.rerank_model = self.model_type.from_pretrained(
                        self.hparams.rerank_name_or_path, cache_dir=self.hparams.cache_dir, return_dict=True
                    )

    def setup(self, mode: str):
        if mode == "fit":
            self.lm_dataset = ConversationalDataset(
                self.hparams.train_dataset_path,
                self.tokenizer,
                max_history=self.hparams.max_history,
                max_seq_length=self.hparams.max_seq_length,
                exclude_kb=self.hparams.exclude_kb,
                include_render=self.hparams.include_render,
            )

            if getattr(self, "nce_dataset", None) is None:
                self.nce_dataset = MaskRefineDataset(
                    self.hparams.train_dataset_path,
                    self.tokenizer,
                    self.mlm_tokenizer,
                    self.ner,
                    self.kge,
                    self.hparams.max_adjacents,
                    self.hparams.max_history,
                    self.hparams.max_seq_length,
                    self.hparams.exclude_kb,
                    self.hparams.include_render,
                    do_lm=not self.hparams.multitask,
                )

    def total_steps(self) -> int:
        num_devices = max(1, self.get_number_of_gpus())  # TODO: consider num_tpu_cores
        effective_batch_size = self.hparams.train_batch_size * self.hparams.accumulate_grad_batches * num_devices

        lm_warmup_steps = (len(self.lm_dataset) / effective_batch_size) * self.hparams.num_lm_warmup_epochs

        if self.hparams.multitask:
            dataset_size = len(self.lm_dataset) + len(self.nce_dataset)
        else:
            dataset_size = len(self.nce_dataset)

        return lm_warmup_steps + (dataset_size / effective_batch_size) * (
            self.hparams.max_epochs - self.hparams.num_lm_warmup_epochs
        )

    def get_dataset(self, mode: str, data_path: str, **kwargs):
        if mode == "test":
            if self.hparams.refine_only:
                test_data_path = self.hparams.generated_path
            else:
                test_data_path = data_path

            return GenPipelineDataset(
                test_data_path,
                self.tokenizer,
                self.halluc_tokenizer,
                max_history=self.hparams.max_history,
                exclude_kb=self.hparams.exclude_kb,
                refine_only=self.hparams.refine_only,
                include_render=self.hparams.include_render,
            )
        elif mode == "refine":
            self.hparams.test_phase = "refine"

            if self.hparams.refine_only and self.hparams.skip_halluc:
                test_data_path = self.hparams.generated_path
            else:
                test_data_path = self._test_output_file("gen")

            return MaskRefineInferenceDataset(
                test_data_path,
                self.tokenizer,
                self.mlm_tokenizer,
                self.ner,
                self.kge,
                self.hparams.max_adjacents,
                self.hparams.max_history,
                self.hparams.max_seq_length,
                self.hparams.exclude_kb,
                self.hparams.pivot_field,
                self.hparams.include_render,
                self.hparams.skip_halluc,
            )
        elif mode == "rerank":
            self.hparams.test_phase = "rerank"

            if self.hparams.rerank_only:
                test_data_path = self.hparams.refined_path
            else:
                test_data_path = self._test_output_file("refine")

            return MaskRerankInferenceDataset(
                test_data_path,
                self.rerank_tokenizer,
                self.hparams.rerank_size,
                self.config.is_encoder_decoder,
                self.hparams.max_history,
                exclude_kb=self.hparams.exclude_kb,
                include_render=self.hparams.include_render,
            )
        elif mode == "valid":
            if self.nce_valid_dataset is None:
                self.nce_valid_dataset = MaskRefineDataset(
                    data_path,
                    self.tokenizer,
                    self.mlm_tokenizer,
                    self.ner,
                    self.kge,
                    self.hparams.max_adjacents,
                    self.hparams.max_history,
                    self.hparams.max_seq_length,
                    self.hparams.exclude_kb,
                    self.hparams.include_render,
                    do_lm=True,
                )
            return self.nce_valid_dataset
        else:
            return self.lm_dataset

    def get_collator(self, mode: str):
        if mode == "test":
            if self.hparams.refine_only:
                return HallucOnlyCollator(
                    self.halluc_tokenizer,
                    self.kge.pad_id,
                    pad_to_multiple_of=self.hparams.pad_to_multiple_of,
                )
            else:
                return GenPipelineCollator(
                    self.tokenizer,
                    self.halluc_tokenizer,
                    self.kge.pad_id,
                    pad_to_multiple_of=self.hparams.pad_to_multiple_of,
                )
        elif mode == "refine":
            return self.get_nce_collator(multiple_pivots_allowed=True)
        elif mode == "rerank":
            return MaskRerankCollator(self.rerank_tokenizer, self.hparams.pad_to_multiple_of)
        elif mode == "valid":
            return self.get_nce_collator()
        else:
            return LMCollator(
                self.tokenizer.pad_token_id,
                self.kge.pad_id,
                as_tuple=True,
                pad_to_multiple_of=self.hparams.pad_to_multiple_of,
            )

    def get_nce_collator(self, multiple_pivots_allowed: bool = False):
        return MaskRefineCollator(
            self.tokenizer,
            self.mlm_tokenizer,
            multiple_pivots_allowed=multiple_pivots_allowed,
            kg_pad=self.kge.pad_id,
            pad_to_multiple_of=self.hparams.pad_to_multiple_of,
        )

    def train_dataloader(self) -> DataLoader:
        if self.current_epoch >= self.hparams.num_lm_warmup_epochs:
            if self.hparams.multitask:
                dataset = MixedDataset(self.lm_dataset, self.nce_dataset)
                collator = MixedCollator(self.get_nce_collator(), self.get_collator("train"))
            else:
                dataset = self.nce_dataset
                collator = self.get_nce_collator()

            return DataLoader(
                dataset,
                batch_size=self.hparams.train_batch_size,
                shuffle=True,
                collate_fn=collator,
                num_workers=self.hparams.num_workers,
                pin_memory=True,
            )
        else:
            return DataLoader(
                self.lm_dataset,
                batch_size=self.hparams.train_batch_size,
                shuffle=True,
                collate_fn=self.get_collator("train"),
                # num_workers=self.hparams.num_workers,
                pin_memory=True,
            )

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, raw_batch: Tuple[Any, ...], batch_nb):
        batch = MultiTaskBatch(raw_batch)
        assert batch.contains_nce or batch.contains_lm

        loss = 0
        if batch.contains_lm:
            lm_output = self(**batch.lm.model_args())

            ppl = torch.clamp(torch.exp(lm_output.loss), max=100, min=0)
            self.log(f"train/lm_ppl", ppl, prog_bar=False)
            self.log(f"ppl", ppl, prog_bar=True, logger=False)

            self.log(f"train/lm_loss", lm_output.loss, prog_bar=False)
            loss += self.hparams.lm_coef * lm_output.loss

        if batch.contains_nce:
            nce_output = self.mask_refine_model(**batch.nce.model_args())
            self.log(f"train/nce_loss", nce_output.loss, prog_bar=False)
            self.log(f"nce_loss", nce_output.loss, logger=False, prog_bar=True)

            self.log(f"train/nce_accuracy", nce_output.accuracy, prog_bar=False)
            self.log(f"nce_acc", nce_output.accuracy, logger=False, prog_bar=True)
            self.log(f"train/mean_reciprocal_rank", nce_output.mean_reciprocal_rank, prog_bar=False)
            self.log(f"train/mean_rank", nce_output.mean_rank, prog_bar=False)
            self.log(f"train/hits1", nce_output.hits1, prog_bar=False)
            self.log(f"train/hits3", nce_output.hits3, prog_bar=False)
            self.log(f"train/hits10", nce_output.hits10, prog_bar=False)

            loss += self.hparams.nce_coef * nce_output.loss
            self._print_selected_candidates(nce_output.selected_cands, batch.nce)

        lr_scheduler = self.trainer.lr_schedulers[0]["scheduler"]
        self.log("train/lr", lr_scheduler.get_last_lr()[-1], prog_bar=False)
        self.log("lr", lr_scheduler.get_last_lr()[-1], prog_bar=True, logger=False)

        self.log(f"train/loss", loss, prog_bar=False)

        return loss

    def _print_selected_candidates(self, selected_cands: torch.Tensor, nce_batch: MaskRefineBatch):
        if not self.hparams.print_selected_cands:
            return

        B, E, C = nce_batch.candidate_ids.shape

        for i in range(B):
            input_seq = self.mlm_tokenizer.decode(
                nce_batch.mlm_inputs["input_ids"][i, :].detach().cpu().tolist(), clean_up_tokenization_spaces=False
            )

            for j in range(E):
                if nce_batch.labels[i, j] < 0:
                    continue

                neighbors = nce_batch.candidate_ids[i, j, :]
                relation_neigbors = nce_batch.candidate_rels[i, j, :]
                selected_cand = self.kge.decode_node(selected_cands[i, j].item())

                label = self.kge.decode_node(nce_batch.labels[i, j].item())
                pivot = self.kge.decode_node(nce_batch.pivot_ids[i, j].item())
                neighbors_list = [self.kge.decode_node(j.item()) for j in neighbors if j.item() > 0]
                relation_list = [self.kge.decode_rel(j.item()) for j in relation_neigbors if j.item() > 0]

                object_rel = [(obj, rel) for obj, rel in zip(neighbors_list, relation_list)]
                print("********** Length Neighbors:{} **********".format(len(neighbors_list)))
                print("***************")
                print(
                    "selected = {} | annotated = {} | pivot = {} | seq = {} | neighbors = {}".format(
                        selected_cand, label, pivot, input_seq, object_rel
                    )
                )
                print("***************")

    def validation_step(self, raw_batch, batch_nb):
        batch = MultiTaskBatch(raw_batch)
        assert batch.contains_nce or batch.contains_lm

        val_output = {}

        # if we dont send labels to model, it doesnt return losses
        if batch.contains_lm:
            lm_output = self(**batch.lm.model_args(exclude_labels=True))
            lm_logits_flat_shifted = lm_output.logits[..., :-1, :].contiguous().view(-1, lm_output.logits.size(-1))
            lm_labels_flat_shifted = batch.lm.labels[..., 1:].contiguous().view(-1)
            loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
            val_loss = loss_fn(lm_logits_flat_shifted, lm_labels_flat_shifted)
            val_output["val_loss"] = val_loss.detach().cpu()

            num_tokens = (batch.lm.labels > 0).int().sum().detach().cpu()
            val_output["num_tokens"] = num_tokens.detach().cpu()

        if batch.contains_nce:
            nce_output = self.mask_refine_model(**batch.nce.model_args(), inbatch_negatives=False)
            val_output["nce_loss"] = nce_output.loss.detach().cpu()
            val_output["nce_accuracy"] = nce_output.accuracy.detach().cpu()
            val_output["mean_reciprocal_rank"] = nce_output.mean_reciprocal_rank.detach().cpu()
            val_output["mean_rank"] = nce_output.mean_rank.detach().cpu()
            val_output["hits1"] = nce_output.hits1.detach().cpu()
            val_output["hits3"] = nce_output.hits3.detach().cpu()
            val_output["hits10"] = nce_output.hits10.detach().cpu()

        return val_output

    def validation_epoch_end(self, outputs: List[Any]):
        val_loss_mean = torch.stack([x["val_loss"] / x["num_tokens"] for x in outputs]).mean().detach().cpu()
        self.log("valid/lm_loss", val_loss_mean, prog_bar=True)
        val_ppl = torch.exp(val_loss_mean)
        self.log("valid/lm_ppl", val_ppl, prog_bar=True)

        for metric in ("nce_loss", "nce_accuracy", "mean_reciprocal_rank", "mean_rank", "hits1", "hits3", "hits10"):
            self.log(
                f"valid/{metric}",
                torch.stack([x[metric] for x in outputs if metric in x]).mean().detach().cpu(),
                prog_bar=False,
            )

        self.log(
            f"valid_nce_acc",
            torch.stack([x["nce_accuracy"] for x in outputs if "nce_accuracy" in x]).mean().detach().cpu(),
            logger=False,
            prog_bar=True,
        )

    def test_step(self, raw_batch, batch_no):
        if getattr(self.hparams, "test_phase", "") == "refine":
            batch = MultiTaskBatch(raw_batch)
            nce_output = self.mask_refine_model(**batch.nce.model_args(), inbatch_negatives=False)

            return {
                "topK_cands": nce_output.topK_cands.detach().cpu().tolist(),
                "topK_rels": nce_output.topK_rels.detach().cpu().tolist(),
                "topK_pivot_ids": nce_output.topK_pivots.detach().cpu().tolist(),
                "topK_pivot_fields": nce_output.topK_pivot_fields.detach().cpu().tolist(),
            }
        elif getattr(self.hparams, "test_phase", "") == "rerank":
            replace_mask = raw_batch.pop("replace_mask")
            if self.config.is_encoder_decoder:
                input_key = "decoder_input_ids"
            else:
                input_key = "input_ids"

            B, E, C, L = replace_mask.shape

            batch = {k: v.view(-1, v.shape[-1]) for k, v in raw_batch.items()}

            rerank_output = self.rerank_model(**batch)
            target_probs = torch.gather(
                F.log_softmax(rerank_output.logits, dim=-1), dim=-1, index=batch[input_key].unsqueeze(-1)
            ).squeeze(-1)
            refine_probs = scatter_sum(target_probs, replace_mask.view(-1, L))[:, 1].reshape(B, E, C)
            reranked_refines = torch.argsort(refine_probs, descending=True)
            return {
                "reranked_refines": reranked_refines.detach().cpu().tolist(),
            }
        else:
            if not self.hparams.refine_only:
                batch = GenPipelineBatch(*raw_batch)
                B = batch.input_ids.shape[0]

                outputs = generate_no_beam_search(
                    self.model,
                    batch.input_ids,
                    token_type_ids=batch.token_type_ids,
                    attention_mask=batch.attention_mask,
                    max_length=self.hparams.max_length,
                    min_length=self.hparams.min_length,
                    do_sample=self.hparams.do_sample,
                    temperature=self.hparams.temperature,
                    top_k=self.hparams.top_k,
                    top_p=self.hparams.top_p,
                    bos_token_id=self.tokenizer.bos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=self.hparams.num_return_sequences,
                    use_cache=True,
                )

                generated_ids = (
                    outputs["generated_ids"].reshape(B, self.hparams.num_return_sequences, -1).detach().cpu().tolist()
                )

                halluc_batch = None
                if not self.hparams.skip_halluc and batch.halluc_input_ids is not None:
                    halluc_input_lengths = (batch.halluc_input_ids != self.halluc_tokenizer.pad_token_id).int().sum(-1)

                    batch_generated_resps = []
                    for i in range(B):
                        halluc_input_ids = batch.halluc_input_ids[i, : halluc_input_lengths[i]].detach().cpu().tolist()
                        for j in range(self.hparams.num_return_sequences):
                            resp = " " + " ".join(
                                [
                                    t.text
                                    for t in self.nlp(
                                        _normalize(
                                            self.tokenizer.decode(generated_ids[i][j], skip_special_tokens=True)
                                        )
                                    )
                                ]
                            )
                            batch_generated_resps.append(
                                halluc_input_ids
                                + self.halluc_tokenizer.encode(resp, add_special_tokens=False)
                                + [self.halluc_tokenizer.eos_token_id]
                            )

                    halluc_batch = self.halluc_tokenizer.pad(
                        dict(input_ids=batch_generated_resps),
                        padding="longest",
                        max_length=self.hparams.max_seq_length,
                        return_tensors="pt",
                        return_attention_mask=True,
                    )
                    halluc_batch = {k: v.to(batch.input_ids.device) for k, v in halluc_batch.items()}
            else:
                halluc_batch, halluc_input_lengths = raw_batch
                B = halluc_input_lengths.shape[0]
                self.hparams.num_return_sequences = halluc_batch["input_ids"].shape[1]
                halluc_batch["input_ids"] = halluc_batch["input_ids"].view(-1, halluc_batch["input_ids"].shape[-1])
                halluc_batch["attention_mask"] = halluc_batch["attention_mask"].view(
                    -1, halluc_batch["attention_mask"].shape[-1]
                )

            if halluc_batch is not None:
                halluc_output = self.halluc_classifier(**halluc_batch)
                halluc_preds = torch.argmax(halluc_output.logits, dim=-1).reshape(
                    B, self.hparams.num_return_sequences, -1
                )

            return [
                {
                    "halluc_prediction": halluc_preds[i, :, halluc_input_lengths[i] :].detach().cpu().tolist()
                    if halluc_batch is not None
                    else None,
                    "generated": None if self.hparams.refine_only else generated_ids[i],
                }
                for i in range(B)
            ]

    @pl.utilities.rank_zero_only
    def test_epoch_end(self, outputs: List[Any]):
        test_phase = getattr(self.hparams, "test_phase", None) or ""

        if test_phase == "refine":
            refine_dataset = self.get_dataset("refine", data_path="")
            self._save_refine_output(refine_dataset.data, outputs)
        elif test_phase == "rerank":
            rerank_dataset = self.get_dataset("rerank", data_path="")
            self._save_rerank_output(rerank_dataset.data, outputs)
        else:
            dataset = self.get_dataset("test", self.hparams.test_dataset_path)
            self._save_gen_output(dataset.data, outputs)

    def _save_rerank_output(self, rerank_data, outputs):
        output_file = self._test_output_file("rerank")
        logger.info(f"reranked output saved in `{output_file}`")

        if self.hparams.rerank_only:
            test_data_path = self.hparams.refined_path
        else:
            test_data_path = self._test_output_file("refine")

        with open(test_data_path, "r", encoding="utf-8") as f:
            original_data = [json.loads(line) for line in f]

        def _rows_equal(d1, d2: Dict[str, Any]) -> bool:
            return (
                d1["dialogue_id"] == d2["dialogue_id"]
                and d1["history"][-self.hparams.max_history :] == d2["history"]
                and d1["response"].strip().split() == d2["response"].strip().split()
            )

        num_rank_changes = 0
        num_total_candidates = 0
        num_samples_with_rank_changes = 0

        with open(output_file, "w") as writer:
            i0, i = 0, 0
            for batch in outputs:
                for reranked_refines in batch["reranked_refines"]:
                    dialogue = rerank_data[i]
                    i += 1

                    while i0 < len(original_data):
                        orig_dialogue = original_data[i0]
                        i0 += 1
                        if _rows_equal(orig_dialogue, dialogue):
                            break

                        orig_dialogue["refines"] = []
                        for gen_response in orig_dialogue["generated_response"]:
                            orig_json = dict(orig_dialogue)
                            orig_json["generated_response"] = gen_response
                            writer.write(json.dumps(orig_json) + "\n")

                    out_json = dict(dialogue)

                    change_found = False
                    for j, refine in enumerate(dialogue["refines"]):
                        safe_reranked_refines = [r for r in reranked_refines[j] if r < len(refine["replacements"])]
                        reranked_replacements = [refine["replacements"][r] for r in safe_reranked_refines]
                        out_json["reranked_replacements"] = reranked_replacements

                        for a, b in zip(safe_reranked_refines, refine["replacements"][: len(safe_reranked_refines)]):
                            if a != b:
                                num_rank_changes += 1
                                change_found = True

                        num_total_candidates += len(reranked_replacements)

                        reranked_refined_resps = [refine["refined_responses"][r] for r in safe_reranked_refines]
                        out_json["reranked_responses"] = reranked_refined_resps

                    if change_found:
                        num_samples_with_rank_changes += 1

                    writer.write(json.dumps(out_json) + "\n")

        while i0 < len(original_data):
            orig_dialogue = original_data[i0]
            orig_dialogue["refines"] = []
            for gen_response in orig_dialogue["generated_response"]:
                orig_json = dict(orig_dialogue)
                orig_json["generated_response"] = gen_response
                writer.write(json.dumps(orig_json) + "\n")
            i0 += 1

        print(
            f"Num of changes due to rerank: {num_rank_changes} out of {num_total_candidates} "
            f"({100. * num_rank_changes / num_total_candidates:.1f}%) corresponding to "
            f"{num_samples_with_rank_changes} ({100. * num_samples_with_rank_changes / len(rerank_data):.1f}%)"
        )

    def merge_test_data_fixed_data(self, test_file, out_file):
        l_responses = []
        fixed_file = self._test_output_file("refine")
        with open(fixed_file, "r") as fixed_test:
            for line in fixed_test.readlines():
                # breakpoint()
                data = json.loads(line)
                l_responses.append(data["response"].strip())

        # breakpoint()
        with open(test_file, "r") as f:
            with open(fixed_file, "r") as fixed:
                with open(out_file, "w") as writer:
                    fixed_lines = fixed.readlines()
                    non_fixed_lines = f.readlines()
                    for fixed, not_fixed in zip(fixed_lines, non_fixed_lines):
                        non_fixed_data = json.loads(not_fixed)
                        response = non_fixed_data["response"].strip()
                        if response not in l_responses:
                            if non_fixed_data["knowledge_base"]:
                                # breakpoint()
                                writer.write(json.dumps(non_fixed_data) + "\n")
                        writer.write(json.dumps(json.loads(fixed)) + "\n")

    def _save_refine_output(self, refine_data, outputs):
        output_file = self._test_output_file("refine")
        logger.info(f"response generation output saved in `{output_file}`")

        if self.hparams.refine_only and self.hparams.skip_halluc:
            gen_data_path = self.hparams.generated_path
        else:
            gen_data_path = self._test_output_file("gen")

        with open(gen_data_path, "r", encoding="utf-8") as f:
            original_data = [json.loads(line) for line in f]

        def _rows_equal(d1: Dict[str, Any], d2) -> bool:
            return (
                d1["dialogue_id"] == d2.dlg_id
                and d1["history"][-self.hparams.max_history :] == d2.history
                and d1["response"].strip().split() == d2.original_response.strip().split()
            )

        with open(output_file, "w") as writer:
            i0, i = 0, 0
            for batch in outputs:
                for top_cands, top_rels, top_pivots, top_pivot_fields in zip(
                    batch["topK_cands"], batch["topK_rels"], batch["topK_pivot_ids"], batch["topK_pivot_fields"]
                ):
                    dialogue = refine_data[i]
                    i += 1

                    while i0 < len(original_data):
                        orig_dialogue = original_data[i0]
                        i0 += 1
                        if _rows_equal(orig_dialogue, dialogue):
                            break

                        orig_dialogue["refines"] = []
                        for gen_response in orig_dialogue["generated_response"]:
                            orig_json = dict(orig_dialogue)
                            orig_json["generated_response"] = gen_response
                            writer.write(json.dumps(orig_json) + "\n")

                    if dialogue.speaker == self.special_tokens.speaker1:
                        speaker = "assistant"
                    else:
                        speaker = "user"

                    out_json = dialogue.json_dict
                    out_json["speaker"] = speaker

                    refines = []
                    for j, (halluc_ent, (pivots, pivot_rel)) in enumerate(dialogue.entities.items()):
                        replacements = [
                            self.kge.decode_node(ent_id) for ent_id in top_cands[j] if ent_id != self.kge.pad_id
                        ]

                        src_triples = []
                        for ent_id, rel_id, pivot_id, pivot_field in zip(
                            top_cands[j], top_rels[j], top_pivots[j], top_pivot_fields[j]
                        ):
                            if ent_id != self.kge.pad_id:
                                ent = self.kge.decode_node(ent_id)
                                rel = self.kge.decode_rel(rel_id)
                                pivot = self.kge.decode_node(pivot_id)
                                if pivot_field == 0:
                                    src_triples.append((pivot, rel, ent))
                                else:
                                    src_triples.append((ent, rel, pivot))

                        refinements = [dialogue.response.replace(halluc_ent, repl) for repl in replacements]

                        refines.append(
                            {
                                "hallucinated_entity": halluc_ent,
                                "pivots": list(pivots.values()),
                                "pivot_rel": pivot_rel,
                                "replacements": replacements,
                                "src_triples": src_triples,
                                "refined_responses": refinements,
                            }
                        )
                    out_json["refines"] = refines

                    writer.write(json.dumps(out_json) + "\n")

            while i0 < len(original_data):
                orig_dialogue = original_data[i0]
                orig_dialogue["refines"] = []
                for gen_response in orig_dialogue["generated_response"]:
                    orig_json = dict(orig_dialogue)
                    orig_json["generated_response"] = gen_response
                    writer.write(json.dumps(orig_json) + "\n")
                i0 += 1
        # merge_file = os.path.join(self.hparams.model_name_or_path, f"merge_refined.jsonl")
        # test_file = self.hparams.generated_path
        # self.merge_test_data_fixed_data(test_file, merge_file)

    def _save_gen_output(self, data, outputs):
        output_file = self._test_output_file("gen")
        logger.info(f"response generation output saved in `{output_file}`")

        label_map = NERLabel.map()

        with open(output_file, "w") as writer:
            i = 0
            for batch in outputs:
                for output in batch:
                    dialogue = data[i]
                    i += 1

                    if dialogue.speaker == self.special_tokens.speaker1:
                        speaker = "assistant"
                    else:
                        speaker = "user"

                    response = getattr(dialogue, "original_response", dialogue.response)

                    out_json = dict(
                        dialogue_id=dialogue.dlg_id,
                        speaker=speaker,
                        history=dialogue.history,
                        response=response,
                        knowledge_base={},
                    )

                    if dialogue.kb_triples:
                        out_json["knowledge_base"]["paths"] = [
                            [triple.subject, triple.predicate, triple.object] for triple in dialogue.kb_triples
                        ]

                    if dialogue.render_kb is not None:
                        out_json["knowledge_base"]["render"] = dialogue.render_kb

                    generated = output["generated"]
                    if generated:
                        out_json["generated_response"] = [
                            self.tokenizer.decode(gen, skip_special_tokens=True).strip() for gen in generated
                        ]
                    else:
                        out_json["generated_response"] = [gen for gen in dialogue.response]

                    halluc_pred = output.get("halluc_prediction", None)
                    if halluc_pred and self.halluc_tokenizer is not None:
                        tokenized_gens = [
                            self.halluc_tokenizer.tokenize(" " + " ".join([t.text for t in self.nlp(g)]))
                            for g in out_json["generated_response"]
                        ]

                        detected_hallucs = []
                        for j, pred in enumerate(halluc_pred):
                            contiguous_seqs = []
                            pred_label = []
                            for r, p in enumerate(pred[: len(tokenized_gens[j])]):
                                p = label_map[p]
                                if p.startswith("B"):
                                    if pred_label:
                                        contiguous_seqs.append([tokenized_gens[j][k] for k in pred_label])
                                    pred_label = [r]
                                elif p.startswith("I"):
                                    if pred_label:
                                        pred_label.append(r)
                                else:
                                    if pred_label:
                                        contiguous_seqs.append([tokenized_gens[j][k] for k in pred_label])
                                    pred_label = []

                            if pred_label:
                                contiguous_seqs.append([tokenized_gens[j][k] for k in pred_label])

                            hallucs = []
                            for seq in contiguous_seqs:
                                halluc_ent = self.halluc_tokenizer.convert_tokens_to_string(seq).strip()
                                if halluc_ent and halluc_ent in out_json["generated_response"][j]:
                                    hallucs.append(halluc_ent)
                            detected_hallucs.append(hallucs)

                        out_json["hallucination_output"] = [
                            [label_map[p] for p in preds[: len(tokenized_gens[j])]] for preds in halluc_pred
                        ]
                        out_json["hallucination_preds"] = detected_hallucs

                    writer.write(json.dumps(out_json) + "\n")

    def _test_output_file(self, mode: str, base_dir: str = None):
        if self.hparams.refine_only:
            output_name = "REFINED_" + Path(self.hparams.generated_path).with_suffix("").name
        elif self.hparams.rerank_only:
            output_name = "RERANKED_" + Path(self.hparams.refined_path).with_suffix("").name
        else:
            output_name = "outputs"
            if mode == "refine":
                output_name += "_refined"
            elif mode == "rerank":
                output_name += "_rerank"
            else:
                output_name += "_gen"

            output_name += f"_maxl{self.hparams.max_length}"
            output_name += f"_hist{self.hparams.max_history}"

            if self.hparams.num_return_sequences > 1:
                output_name += f"_num{self.hparams.num_return_sequences}"
            if self.hparams.do_sample:
                if self.hparams.top_k > 0:
                    output_name += f"_topk{self.hparams.top_k}"
                if self.hparams.top_p > 0:
                    output_name += f"_topp{self.hparams.top_p}"
            else:
                output_name += f"_greedy"
            if self.hparams.temperature != 1.0:
                output_name += f"_temp{self.hparams.temperature}"
            if self.hparams.repetition_penalty != 1.0:
                output_name += f"_repp{self.hparams.repetition_penalty}"

            if self.hparams.pivot_field == "subject+object":
                output_name += "_pivotSUBJ-OBJ"
            else:
                output_name += f"_pivot{self.hparams.pivot_field.upper()}"

        return os.path.join(base_dir or self.hparams.model_name_or_path, f"{output_name}.jsonl")

    @pl.utilities.rank_zero_only
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        save_path = self.output_dir.joinpath("best_model")

        self.model.config.save_step = self.step_count
        lm_save_path = save_path.joinpath("lm")
        self.model.save_pretrained(lm_save_path)
        self.tokenizer.save_pretrained(lm_save_path)

        if self.mlm_model is not None:
            mlm_save_path = save_path.joinpath("mlm")
            self.mlm_model.config.save_step = self.step_count
            self.mlm_model.save_pretrained(mlm_save_path)
            self.mlm_tokenizer.save_pretrained(mlm_save_path)

        self.mask_refine_model.save_pretrained(save_path)

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        BaseTransformer.add_model_specific_args(parser, root_dir)
        BaseTransformer.add_generation_args(parser)

        parser.add_argument(
            "--mlm_model",
            default="roberta-large",
            type=str,
            help="Path to pretrained model or model identifier from huggingface.co/models",
        )

        parser.add_argument(
            "--halluc_classifier",
            default=None,
            type=str,
            help="Path to pretrained model for detecting hallucination, used only during inference",
        )

        parser.add_argument(
            "--kge",
            default=None,
            type=str,
            help="Path to a directory containing KG embedding files",
        )
        parser.add_argument(
            "--graph_file", type=str, required=True, help="Path to the graph file consisting of triples per line"
        )

        parser.add_argument(
            "--exclude_kb",
            action="store_true",
            default=False,
            help="Whether to exclude knowledge from input sequences.",
        )

        parser.add_argument(
            "--include_render",
            action="store_true",
            default=False,
            help="Whether to include render in input sequences.",
        )

        parser.add_argument(
            "--bypass_mlm",
            action="store_true",
            default=False,
            help="Whether to bypass MLM during training.",
        )

        parser.add_argument(
            "--max_adjacents",
            default=0,
            type=int,
            help="Maximum number of adjacent nodes for NCE computations (0 denotes no constraint)",
        )

        parser.add_argument(
            "--num_lm_warmup_epochs",
            default=0,
            type=int,
            help="Number of LM epochs (used when `--mt_strategy` is set to alternate)",
        )

        parser.add_argument(
            "--multitask",
            default=False,
            action="store_true",
            help="Whether to train in a multi-task fashion",
        )

        parser.add_argument(
            "--lm_coef",
            default=1.0,
            type=float,
            help="Language model loss coefficient",
        )
        parser.add_argument(
            "--nce_coef",
            default=1.0,
            type=float,
            help="NCE loss coefficient",
        )
        parser.add_argument(
            "--no_stabilize",
            default=False,
            action="store_true",
            help="Disable stabilizing scores for NCE loss computation",
        )
        parser.add_argument(
            "--inbatch_negatives",
            default=False,
            action="store_true",
            help="Whether to include in-batch negative examples for NCE loss computation",
        )
        parser.add_argument(
            "--print_selected_cands",
            default=False,
            action="store_true",
            help="Whether to print selected candidates",
        )

        parser.add_argument(
            "--pivot_field",
            type=str,
            default="subject+object",
            choices=("subject", "object", "subject+object"),
            help="During inference, the triple field to use as pivot for refining hallucinations",
        )

        parser.add_argument(
            "--refine_only",
            default=False,
            action="store_true",
            help="During inference, skips generation and perform only refinement",
        )
        parser.add_argument(
            "--generated_path",
            type=str,
            default=None,
            help="During inference, path to the generated response file used when `--refine_only` is set",
        )
        parser.add_argument(
            "--rerank_name_or_path",
            type=str,
            default=None,
            help="Path to the rerank model (it can be a T5 or GPT-2), reranking ignored if not set",
        )
        parser.add_argument(
            "--rerank_only",
            default=False,
            action="store_true",
            help="During inference, skips generation and refinement and performs only reranking using T5",
        )
        parser.add_argument(
            "--rerank_size",
            default=8,
            type=int,
            help="Number of top refinements to rerank",
        )
        parser.add_argument(
            "--refined_path",
            type=str,
            default=None,
            help="During inference, path to the refined response file used when `--rerank_only` is set",
        )

        parser.add_argument(
            "--skip_halluc",
            action="store_true",
            default=False,
            help="Whether to skip the hallucination classifier phase during inference and instead, use all entities.",
        )


def _validate(args):
    if args.do_generate:
        if args.refine_only and args.rerank_only:
            raise ValueError("`--refine_only` and `--rerank_only` cannot be set at the same time")

        if not args.refine_only and not args.rerank_only and args.test_dataset_path is None:
            raise ValueError("`--test_dataset_path` is required when `--do_generate` is set")

        if args.refine_only and args.generated_path is None:
            raise ValueError("when `--refine_only` is set, `--generated_path` is required")

        if args.rerank_only and args.refined_path is None:
            raise ValueError("when `--rerank_only` is set, `--refined_path` is required")


def main():
    parser = ArgumentParser()

    add_generic_args(parser)
    DialogueMaskRefineTransformer.add_model_specific_args(parser, os.getcwd())
    args = parser.parse_args()
    _validate(args)

    odir = Path(args.output_dir)
    if args.overwrite_output_dir and odir.exists():
        shutil.rmtree(odir)
    odir.mkdir(parents=True, exist_ok=True)

    model = DialogueMaskRefineTransformer(args)

    extra_callbacks = []
    if args.do_train and args.patience > 0:
        extra_callbacks.append(
            pl.callbacks.EarlyStopping(
                monitor="valid/lm_ppl",
                # monitor="valid/hits1",
                min_delta=args.min_delta,
                patience=args.patience,
                verbose=False,
                mode="min",
            )
        )

    if args.wandb and is_wandb_available():
        pl_logger = pl_loggers.WandbLogger(
            project="DialogueMaskRefineKB",
            name="Finetune-{}".format(args.namestr),
            anonymous=not authorize_wandb(),
        )
    else:
        pl_logger = pl_loggers.TensorBoardLogger(
            save_dir=args.output_dir,
            name="train_logs" if args.do_train else "valid_logs",
            default_hp_metric=False,
        )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.output_dir,
        prefix="checkpoint",
        # monitor="valid/hits1",
        monitor="valid/lm_ppl",
        mode="min",
        save_top_k=1,
    )

    trainer_kwargs = {
        "reload_dataloaders_every_epoch": True,
        "weights_summary": "top",
    }

    trainer = generic_train(
        model,
        args,
        pl_logger,
        extra_callbacks,
        checkpoint_callback,
        **trainer_kwargs,
    )

    if args.do_generate:
        if args.do_train:
            logger.info("Reloading best model after training...")
            model = DialogueMaskRefineTransformer(args)

        if not args.rerank_only:
            if not args.refine_only or not args.skip_halluc:
                logger.info(
                    f"Multi-stage generation pipeline: (1) {'' if args.refine_only else 'response generation and '}"
                    f"detecting hallucinations"
                )
                trainer.test(model)

            logger.info("Multi-stage generation pipeline: (2) refine hallucinations")
            trainer.test(
                model,
                test_dataloaders=model.get_dataloader(
                    "refine", args.eval_batch_size, data_path=args.test_dataset_path
                ),
            )

        if args.rerank_name_or_path is not None:
            logger.info("Multi-stage generation pipeline: (3) rerank refinements")
            trainer.test(
                model,
                test_dataloaders=model.get_dataloader(
                    "rerank", args.eval_batch_size, data_path=args.test_dataset_path
                ),
            )
