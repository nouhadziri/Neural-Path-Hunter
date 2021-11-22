import json
import logging
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import spacy
import torch
from torch.utils.data import Dataset
from transformers import (
    PreTrainedTokenizerBase,
    PreTrainedTokenizer,
    PreTrainedModel,
)
from .graph import refine_node
from .kge import KnowledgeGraphEmbedding

logger = logging.getLogger("conv_data")

TokenizedText = List[str]
EncodedText = List[int]

SPECIAL_TOKENS = [
    "<bos>",
    "<eos>",
    "<pad>",
    "<speaker1>",
    "<speaker2>",
    "<subject>",
    "<predicate>",
    "<object>",
    "<triple>",
    "<sep>",
]

ATTR_TO_SPECIAL_TOKEN = {
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "pad_token": "<pad>",
    "sep_token": "<sep>",
    "additional_special_tokens": [
        "<speaker1>",
        "<speaker2>",
        "<subject>",
        "<predicate>",
        "<object>",
        "<triple>",
    ],
}


def add_special_tokens(tokenizer: PreTrainedTokenizer, model: PreTrainedModel):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    orig_num_tokens = len(tokenizer.encoder)
    num_added_tokens = tokenizer.add_special_tokens(
        ATTR_TO_SPECIAL_TOKEN
    )  # returns 0 and doesn't add if they are already there

    if num_added_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)


@dataclass
class Triple:
    subject: Union[str, EncodedText]
    predicate: Union[str, EncodedText]
    object: Union[str, EncodedText]

    def encode(
        self, tokenizer: PreTrainedTokenizerBase, sep_token_id: Optional[int] = None, add_sep: bool = True
    ) -> EncodedText:
        if add_sep:
            sep_token_id = sep_token_id or tokenizer.sep_token_id

        return (
            self._may_encode("subject", tokenizer)
            + ([sep_token_id] if add_sep else [])
            + self._may_encode("predicate", tokenizer)
            + ([sep_token_id] if add_sep else [])
            + self._may_encode("object", tokenizer)
        )

    def _may_encode(self, attr: str, tokenizer: PreTrainedTokenizerBase) -> EncodedText:
        val = getattr(self, attr, None)
        if val is None:
            return []
        elif isinstance(val, str):
            return tokenizer.encode(val, add_special_tokens=False)
        else:
            return val

    @classmethod
    def of(cls, triples) -> Iterable["Triple"]:
        for triple in triples:
            if len(triple) < 3:
                continue

            s, p, o = refine_node(triple[0]), triple[1], refine_node(triple[2])
            if not s or not p or not o:
                continue

            if p.startswith("~"):
                p = p[1:]
                s, o = o, s

            yield Triple(s, p, o)


@dataclass
class SpecialTokens:
    bos: int
    eos: int
    pad: int
    speaker1: int
    speaker2: int
    subject: int
    predicate: int
    object: int
    triple: int
    sep: int


@dataclass
class Example:
    dlg_id: int
    history: List[str]
    speaker: Union[int, str]
    response: str
    kb_triples: List[Triple]
    render_kb: Optional[str]

    def __len__(self):
        return (
            sum(len(t) for t in self.history)
            + len(self.response)
            + (len(self.render_kb) if self.render_kb is not None else 0)
        )

    @property
    def json_dict(self) -> Dict[str, Any]:
        out_json = dict(
            dialogue_id=self.dlg_id,
            speaker=self.speaker,
            history=self.history,
            response=self.response,
            knowledge_base={},
        )

        if self.kb_triples:
            out_json["knowledge_base"]["paths"] = [
                [triple.subject, triple.predicate, triple.object] for triple in self.kb_triples
            ]

        if self.render_kb is not None:
            out_json["knowledge_base"]["render"] = self.render_kb

        return out_json


class ConversationalDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        tokenizer: PreTrainedTokenizerBase,
        kge: Optional[KnowledgeGraphEmbedding] = None,
        max_history: int = 1,
        max_seq_length: int = 0,
        exclude_kb: bool = False,
        mmi: bool = False,
        include_history: bool = False,
        is_generation: bool = False,
        spacy_tokenize: bool = False,
        include_render: bool = False,
    ):
        self.tokenizer = tokenizer
        self.kge = kge
        self.special_tokens = SpecialTokens(*tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS))
        self.max_history = max_history
        self.max_seq_length = max_seq_length
        self.exclude_kb = exclude_kb
        self.include_render = include_render
        self.mmi = mmi
        self.include_history = include_history
        self.is_generation = is_generation
        self.spacy_tokenize = spacy_tokenize

        if self.spacy_tokenize:
            self.nlp = spacy.load("en_core_web_sm", disable=("ner", "parser", "tagger", "lemmatizer"))
        else:
            self.nlp = None

        with open(dataset_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]

        self.data = list(self._flatten(data))

        if len(self.data) < len(data):
            logger.warning(
                f"data at `{dataset_path}` got shrunk due to max length constraint: "
                f"{len(data)} -> {len(self.data)} = {len(data) - len(self.data)} examples "
                f"~{100 * (1 - len(self.data) / len(data)):.1f}% reduction"
            )

    def __len__(self) -> int:
        return len(self.data)

    # flatten
    def _flatten(self, data: List[Dict[str, Any]]) -> Iterable[Example]:
        """
        Flattens dialogues so that each predictable turn becomes a top level entry.
        """
        num_no_triple = 0
        num_incomplete_triple = 0

        for dialogue in data:
            history = dialogue["history"][-self.max_history :]
            # breakpoint()
            speaker = dialogue["speaker"]
            if speaker.lower() in ("user", "wizard"):
                speaker = self.special_tokens.speaker2
            else:
                speaker = self.special_tokens.speaker1
            if self.max_history > 0:
                history = history[-self.max_history :]
            response = dialogue["response"]
            if dialogue["knowledge_base"]:
                kb_triples = list(Triple.of(dialogue["knowledge_base"]["paths"]))
                if len(kb_triples) < len(dialogue["knowledge_base"]["paths"]):
                    num_incomplete_triple += 1
                render_kb = dialogue["knowledge_base"]["render"]
            else:
                num_no_triple += 1
                kb_triples = []
                render_kb = None
            dialogue_id = dialogue["dialogue_id"]

            if self.max_seq_length > 0:
                encoded_len = (
                    len(kb_triples) * 6
                    + len(self.tokenizer.tokenize(" ".join(history) + " " + response))
                    + len(history)
                    + 1
                    + 2
                )
                if encoded_len >= self.max_seq_length:
                    continue

            yield Example(
                dlg_id=dialogue_id,
                history=history,
                speaker=speaker,
                response=response,
                kb_triples=kb_triples,
                render_kb=render_kb,
            )

        num_with_triple = len(data) - num_no_triple - num_incomplete_triple
        logger.info(
            "#examples with triples {} ({:.1f}%) out of {} examples".format(
                num_with_triple, 100 * num_with_triple / len(data), len(data)
            )
        )

        logger.info(
            "#examples with no triples {} ({:.1f}%) and incomplete triples {} ({:.1f}%)".format(
                num_no_triple,
                100 * num_no_triple / len(data),
                num_incomplete_triple,
                100 * num_incomplete_triple / len(data),
            )
        )

    def _word_tokenize(self, text: str, *, as_string: bool = True) -> Union[str, List[str]]:
        assert self.spacy_tokenize

        # To resolve issues like 'He also directed Batman R.I.P.. Have you seen that one?'
        text = re.sub(r"(\w+\.)\.\s", r"\1 . ", text)
        text = re.sub(r"(\.\w\.)\.\s", r"\1 . ", text)

        # To resolve issues like 'I like Neil Brown Jr..' and 'I think he plays for Real Madrid C.F..'
        if re.match(r".*\w+\.\.$", text) or re.match(r".*\.\w\.\.$", text):
            text = text[:-1] + " ."

        tokens = [tok.text for tok in self.nlp(text)]
        if as_string:
            return " ".join(tokens)
        else:
            return tokens

    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self.data[index]

        if self.spacy_tokenize:
            response = self._word_tokenize(" " + example.response)
            history = [self._word_tokenize(u) for u in example.history]
        else:
            response = example.response
            history = example.history

        history = [self.tokenizer.encode(h) for h in history]
        response = self.tokenizer.encode(response)
        triples = []
        if not self.exclude_kb:
            for triple in example.kb_triples:
                if self.kge is None:
                    encoded_triple = (
                        self.tokenizer.encode(triple.subject),
                        self.tokenizer.encode(triple.predicate),
                        self.tokenizer.encode(triple.object),
                    )
                else:
                    encoded_triple = (
                        self.kge.encode_node(triple.subject),
                        self.kge.encode_rel(triple.predicate),
                        self.kge.encode_node(triple.object),
                    )
                triples.append(Triple(*encoded_triple))

        render = None
        if self.include_render and example.render_kb is not None:
            render = self.tokenizer.encode(example.render_kb, add_special_tokens=False)

        if self.mmi:
            item_dict = self._build_triple_from_response(
                triples,
                render,
                history,
                example.speaker,
                response,
                with_eos=not self.is_generation,
                include_history=self.include_history,
            )
        else:
            item_dict = self._build_from_segments(
                triples, render, history, example.speaker, response, with_eos=not self.is_generation
            )

        if self.kge is not None:
            item_dict["subject_ids"] = [t.subject for t in triples]
            item_dict["predicate_ids"] = [t.predicate for t in triples]
            item_dict["object_ids"] = [t.object for t in triples]

        return item_dict

    def _build_from_segments(
        self,
        kb_triples: List[Triple],
        render: Optional[List[int]],
        history: List[List[int]],
        speaker: int,
        response: List[int],
        with_eos: bool = True,
    ) -> Dict[str, List[int]]:
        """ Builds a sequence of input from 3 segments: history, kb triples and response. """

        tokens, token_types = [self.special_tokens.bos], [self.special_tokens.bos]
        # KB
        token_ids_triples = []
        token_types_triples = []
        if kb_triples is not None:
            for triple in kb_triples:
                if self.kge is None:
                    token_ids_triples.extend(
                        [self.special_tokens.subject]
                        + triple.subject
                        + [self.special_tokens.predicate]
                        + triple.predicate
                        + [self.special_tokens.object]
                        + triple.object
                    )
                    token_types_triples.extend(
                        [self.special_tokens.subject] * (len(triple.subject) + 1)
                        + [self.special_tokens.predicate] * (len(triple.predicate) + 1)
                        + [self.special_tokens.object] * (len(triple.object) + 1)
                    )
                else:
                    token_ids_triples.extend([self.tokenizer.pad_token_id] * 3)
                    token_types_triples.extend(
                        [self.special_tokens.subject, self.special_tokens.predicate, self.special_tokens.object]
                    )

        if render:
            token_ids_render = render
            token_types_render = [self.special_tokens.triple] * len(render)
        else:
            token_ids_render = []
            token_types_render = []

        # History
        token_id_history = []
        token_type_history = []
        sequence = history + ([response] if with_eos else [])

        if len(history) % 2 == 1:
            current_speaker = self._other_speaker(speaker)
        else:
            current_speaker = speaker

        for i, utterance in enumerate(sequence):
            token_id_history.extend([current_speaker] + utterance)
            token_type_history.extend([current_speaker] * (len(utterance) + 1))
            current_speaker = self._other_speaker(current_speaker)

        tokens.extend(token_ids_triples + token_ids_render + token_id_history)
        token_types.extend(token_types_triples + token_types_render + token_type_history)

        # For training
        if with_eos:
            tokens.append(self.special_tokens.eos)
            token_types.append(self._other_speaker(current_speaker))

            labels = {
                "lm_labels": (
                    [-100]
                    + [-100] * (len(token_ids_triples) + len(token_id_history) - len(response))
                    + response
                    + [self.special_tokens.eos]
                )
            }

        # For testing
        else:
            labels = {}
            tokens.append(current_speaker)
            token_types.append(current_speaker)

        return dict(
            input_ids=tokens,
            token_type_ids=token_types,
            attention_mask=[1] * len(tokens),
            triple_ids=token_ids_triples,
            triple_type_ids=token_types_triples,
            **labels,
            # current_speaker=current_speaker,
        )

    def _build_triple_from_response(
        self,
        kb_triples: List[Triple],
        render: Optional[List[int]],
        history: List[List[int]],
        speaker: int,
        response: List[int],
        with_eos: bool = True,
        include_history: bool = True,
    ) -> Dict[str, List[int]]:
        """ history + response -> triple"""
        token_ids_triples = []
        token_types_triples = []
        tokens, token_types = [self.special_tokens.triple], [self.special_tokens.triple]
        if kb_triples is not None:
            for triple in kb_triples:
                if self.kge is None:
                    token_ids_triples.extend(
                        [self.special_tokens.subject]
                        + triple.subject
                        + [self.special_tokens.predicate]
                        + triple.predicate
                        + [self.special_tokens.object]
                        + triple.object
                    )
                    token_types_triples.extend(
                        [self.special_tokens.subject] * (len(triple.subject) + 1)
                        + [self.special_tokens.predicate] * (len(triple.predicate) + 1)
                        + [self.special_tokens.object] * (len(triple.object) + 1)
                    )

        token_ids_render = [] or render
        if token_ids_render:
            token_types_render = [self.special_tokens.triple] * len(render)
        else:
            token_types_render = []

        token_id_history = []
        token_type_history = []

        sequence = (history if include_history else []) + [response]
        # breakpoint()
        # TODO check whether speaker is correct if there is include_history = false
        if len(history) % 2 == 1:
            current_speaker = self._other_speaker(speaker)
        else:
            current_speaker = speaker

        for utterance in sequence:
            # breakpoint()
            token_id_history.extend([current_speaker] + utterance)
            token_type_history.extend([current_speaker] * (len(utterance) + 1))
            current_speaker = self._other_speaker(current_speaker)

        tokens.extend(token_id_history + token_ids_triples + token_ids_render)
        token_types.extend(token_type_history + token_types_triples + token_types_render)

        if with_eos:
            tokens.append(self.special_tokens.eos)
            token_types.append(self._other_speaker(current_speaker))

            labels = {
                "lm_labels": ([-100] + [-100] * len(token_id_history) + token_ids_triples + [self.special_tokens.eos])
            }
        else:
            labels = {}
            tokens.append(current_speaker)
            token_types.append(current_speaker)

        return dict(
            input_ids=tokens,
            token_type_ids=token_types,
            **labels,
        )

    def _other_speaker(self, speaker: int):
        if speaker == self.special_tokens.speaker1:
            return self.special_tokens.speaker2
        else:
            return self.special_tokens.speaker1


@dataclass
class LMBatch:
    input_ids: torch.LongTensor
    token_type_ids: torch.LongTensor
    attention_mask: torch.LongTensor
    triple_ids: torch.LongTensor
    triple_type_ids: torch.LongTensor
    labels: Optional[torch.LongTensor] = None
    subject_ids: Optional[torch.LongTensor] = None
    predicate_ids: Optional[torch.LongTensor] = None
    object_ids: Optional[torch.LongTensor] = None

    def model_args(self, include_nones=False, exclude_labels=False):
        return {
            k: v
            for k, v in asdict(self).items()
            if (include_nones or v is not None)
            and (not exclude_labels or k != "labels")
            and k not in ("triple_ids", "triple_type_ids")
        }


@dataclass
class Collator:
    pad: int
    kg_pad: Optional[int] = None
    mask_pad: Optional[int] = 0
    label_pad: Optional[int] = -100
    as_tuple: bool = False
    fields: Iterable[str] = (
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "triple_ids",
        "triple_type_ids",
        "lm_labels",
        "subject_ids",
        "predicate_ids",
        "object_ids",
    )
    pad_to_multiple_of: Optional[int] = None

    @property
    def special_pad_tokens(self) -> Dict[str, int]:
        return dict(
            attention_mask=self.mask_pad,
            lm_labels=self.label_pad,
            subject_ids=self.kg_pad,
            predicte_ids=self.kg_pad,
            object_ids=self.kg_pad,
        )

    def _get_pad_token(self, field: str) -> int:
        pad = self.special_pad_tokens.get(field, None)
        return self.pad if pad is None else pad

    def __call__(self, batch):
        """Padding the instances within each batch. Batch is whatever you get from __get_item__().
        Batch is a list of dictionaries.
        """
        bsz = len(batch)
        padded_batch = {}
        # Go over each data point in the batch and pad each example.
        for name in self.fields:
            if any(name not in x for x in batch):
                continue

            max_l = max(len(x[name]) for x in batch)
            if self.pad_to_multiple_of:
                max_l = int(self.pad_to_multiple_of * np.ceil(max_l / self.pad_to_multiple_of))

            pad_token = self._get_pad_token(name)

            # Fill the batches with padding tokens
            padded_field = np.full((bsz, max_l), pad_token, dtype=np.int64)

            # batch is a list of dictionaries
            for bidx, x in enumerate(batch):
                padded_field[bidx, : len(x[name])] = x[name]

            padded_batch[name] = torch.from_numpy(padded_field)


        if self.as_tuple:
            return tuple(padded_batch.get(f, None) for f in self.fields)
        else:
            return padded_batch
