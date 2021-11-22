import json
import logging
import re
from collections import OrderedDict
from dataclasses import dataclass, asdict
from itertools import takewhile
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np
import spacy
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    PreTrainedTokenizerBase,
)

from ..conv_data import (
    ConversationalDataset,
    Collator as LMCollator,
    LMBatch,
    Triple,
    Example,
    SpecialTokens,
    SPECIAL_TOKENS,
    TokenizedText,
    EncodedText,
)
from ..graph import EntityTagger, load_kg
from ..kge import KnowledgeGraphEmbedding

logger = logging.getLogger("mask_refine_data")
MT_STRATEGIES = ("mix", "alternate")


def _normalize(text: str) -> str:
    return " ".join(text.split())


@dataclass
class DialogueEntityExample(Example):
    entities: Optional[Dict[str, Tuple[str, str, str]]] = None


@dataclass
class DialogueInferenceExample(Example):
    original_response: str
    response_idx: int = 0
    entities: Optional[Dict[str, Tuple[Dict[str, str], str]]] = None

    @property
    def json_dict(self) -> Dict[str, Any]:
        out_json = dict(
            dialogue_id=self.dlg_id,
            speaker=self.speaker,
            history=self.history,
            response=self.original_response,
            generated_response=self.response,
            knowledge_base={},
        )

        if self.kb_triples:
            out_json["knowledge_base"]["paths"] = [
                [triple.subject, triple.predicate, triple.object] for triple in self.kb_triples
            ]

        if self.render_kb is not None:
            out_json["knowledge_base"]["render"] = self.render_kb

        return out_json


class NER:
    def __init__(self, output_path, method: str = "kg", graph_file: str = None, knowledge_graph: nx.Graph = None):
        logger.info(f"loading NER model using {method}...")
        if knowledge_graph is None:
            assert graph_file is not None
            self.knowledge_graph = load_kg(graph_file)
        else:
            self.knowledge_graph = knowledge_graph

        if method == "spacy":
            self._nlp = spacy.load("en_core_web_sm")
        elif method == "kg":
            self._graph_tagger = EntityTagger(self.knowledge_graph, output_path)
        else:
            raise ValueError(f"Unknown NER method: `{method}`")
        self.method = method

    def extract(self, text: str) -> List[Tuple[str, str, Optional[str]]]:
        if self.method == "spacy":
            return self._spacy_extract(text)
        else:
            return self._graph_extract(text)

    def _spacy_extract(self, text: str) -> List[Tuple[str, str, str]]:
        tokens = self._nlp(text)

        ents = []

        ent, ent_type = [], []
        start_offset, end_offset = 0, 0
        for tok in tokens:
            if tok.ent_iob_ == "B":
                if ent:
                    if text[start_offset:end_offset] in self.knowledge_graph:
                        ents.append((" ".join(ent), ent_type[0]))
                ent.append(tok.text)
                ent_type.append(tok.ent_type_)
                start_offset = tok.idx
                end_offset = start_offset + len(tok.text)
            elif tok.ent_iob_ == "I":
                ent.append(tok.text)
                ent_type.append(tok.ent_type_)
                end_offset = tok.idx + len(tok.text)

        if ent:
            if text[start_offset:end_offset] in self.knowledge_graph:
                ents.append((" ".join(ent), text[start_offset:end_offset], ent_type[0]))

        return ents

    def _graph_extract(self, text: str) -> List[Tuple[str, str, Optional[str]]]:
        return [(ent, kg_key, None) for ent, kg_key in self._graph_tagger.extract(text)]


@dataclass
class MaskRefineBatch:
    mlm_inputs: Union[Dict[str, torch.LongTensor], torch.LongTensor]
    mlm_entity_mask: Optional[torch.LongTensor]
    lm_input_ids: torch.LongTensor
    lm_attention_masks: torch.LongTensor
    candidate_ids: torch.LongTensor
    candidate_rels: torch.LongTensor
    pivot_ids: torch.LongTensor
    pivot_fields: Optional[torch.LongTensor] = None
    labels: Optional[torch.LongTensor] = None
    label_indices: Optional[torch.LongTensor] = None

    def model_args(self, include_nones=False) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if include_nones or v is not None}


@dataclass
class GenPipelineBatch:
    input_ids: torch.LongTensor
    token_type_ids: torch.LongTensor
    attention_mask: torch.LongTensor
    halluc_input_ids: torch.LongTensor
    halluc_input_lengths: Optional[torch.LongTensor] = None


class MultiTaskBatch:
    def __init__(self, raw_batch):
        self.nce = None
        self.lm = None

        if len(raw_batch) == 1:
            self.nce = MaskRefineBatch(*raw_batch[0])
        elif len(raw_batch) == 2:
            if raw_batch[0] is not None:
                self.nce = MaskRefineBatch(*raw_batch[0])
            if raw_batch[1] is not None:
                self.lm = LMBatch(*raw_batch[1])
        else:
            self.lm = LMBatch(*raw_batch)

    @property
    def contains_nce(self) -> bool:
        return self.nce is not None

    @property
    def contains_lm(self) -> bool:
        return self.lm is not None


@dataclass
class MLMExample:
    inputs: Dict[str, EncodedText]
    response_entity_mask: List[int]
    entity_mask: List[int]


@dataclass
class LMExample:
    inputs: Dict[str, EncodedText]
    attention_masks: List[EncodedText]


@dataclass
class MaskRefineExample:
    mlm_inputs: Union[Dict[str, EncodedText], List[int]]
    mlm_entity_mask: Optional[List[int]]
    inputs: Dict[str, EncodedText]
    attention_masks: List[EncodedText]
    candidate_ids: Union[List[List[int]], List[List[List[int]]]]
    candidate_rels: Union[List[List[int]], List[List[List[int]]]]
    pivot_ids: Union[List[int], List[List[int]]]
    pivot_fields: Union[List[int], List[List[int]]]
    labels: Optional[List[int]] = None
    label_indices: Optional[List[int]] = None

    @property
    def has_mlm(self) -> bool:
        return self.mlm_inputs is not None


@dataclass
class CombinedExample:
    lm_inputs: Optional[Dict[str, List[Any]]] = None
    mask_refine: Optional[MaskRefineExample] = None
    rerank_inputs: Optional[Dict[str, List[Any]]] = None

    @property
    def has_lm(self):
        return self.lm_inputs is not None

    @property
    def has_mask_refine(self):
        return self.mask_refine is not None


class MaskRefineDataset(ConversationalDataset):
    def __init__(
        self,
        dataset_path: str,
        lm_tokenizer: PreTrainedTokenizerBase,
        mlm_tokenizer: Optional[PreTrainedTokenizerBase],
        ner: NER,
        kge: KnowledgeGraphEmbedding,
        max_adjacents: int = 0,
        max_history: int = 1,
        max_seq_length: int = 0,
        exclude_kb: bool = False,
        include_render: bool = False,
        do_lm: bool = True,
    ):
        self.mlm_tokenizer = mlm_tokenizer
        self.max_adjacents = max_adjacents
        self.do_lm = do_lm
        if self.mlm_tokenizer is not None:
            self.mlm_special_tokens = SpecialTokens(*self.mlm_tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS))
        else:
            self.mlm_special_tokens = None

        self.ner = ner

        super().__init__(
            dataset_path,
            lm_tokenizer,
            kge=kge,
            max_history=max_history,
            max_seq_length=max_seq_length,
            exclude_kb=exclude_kb,
            spacy_tokenize=True,
            include_render=include_render,
        )

    def __len__(self) -> int:
        return len(self.data)

    # flatten
    def _flatten(self, data: List[Dict[str, Any]]) -> Iterable[DialogueEntityExample]:
        """
        Flattens dialogues so that each predictable turn becomes a top level entry.
        """
        new_nodes, new_rels = set(), set()
        num_notfound_triples = 0
        num_nce_instances = 0
        num_lm_instances = 0

        for dialogue in tqdm(data, desc="Reading data for mask refine"):
            response = _normalize(dialogue["response"])

            if dialogue["knowledge_base"]:
                # kb_triples = list(Triple.of(dialogue["knowledge_base"]))
                kb_triples = list(Triple.of(dialogue["knowledge_base"]["paths"]))
                # for triple in kb_triples:
                #     sub, obj = triple[0], triple[2]
                #     if sub == "" or obj == "":
                if kb_triples:
                    render_kb = dialogue["knowledge_base"]["render"]
                else:
                    continue
            else:
                continue

            dialogue_id = dialogue["dialogue_id"]
            speaker = dialogue["speaker"]

            if speaker.lower() in ("user", "wizard"):
                speaker = self.special_tokens.speaker2
            else:
                speaker = self.special_tokens.speaker1

            history = dialogue["history"]
            if self.max_history > 0:
                history = history[-self.max_history :]

            if self.max_seq_length > 0:
                encoded_len = (
                    sum(len(t.encode(self.tokenizer)) + 1 for t in kb_triples)
                    + len(self.tokenizer.tokenize(" ".join(history) + " " + response))
                    + len(history)  # speakers
                    + 1  # BOS
                    + 1  # EOS
                )
                if encoded_len >= self.max_seq_length:
                    continue

            if kb_triples:
                entity_refs, n_notfounds = self._build_entity_ref(response, kb_triples, new_nodes, new_rels)
                num_notfound_triples += n_notfounds
            else:
                entity_refs = None

            if entity_refs:
                num_nce_instances += 1
            elif not self.do_lm:
                continue

            num_lm_instances += 1

            yield DialogueEntityExample(
                dlg_id=dialogue_id,
                history=history,
                speaker=speaker,
                response=response,
                kb_triples=kb_triples,
                render_kb=render_kb,
                entities=entity_refs,
            )

        logger.info(
            f"Number of instances used for LM training: {num_lm_instances} ({100. * num_lm_instances / len(data):.1f}%)"
        )
        logger.info(
            f"Number of instances used for NCE training: "
            f"{num_nce_instances} ({100. * num_nce_instances / num_lm_instances:.1f}%) - "
            f"of total data {100 * num_nce_instances / len(data):.1f}%"
        )

        logger.info(f"Number of triples that do not exist in KG: {num_notfound_triples}")
        logger.info(f"Randomly initializing embeddings for {len(new_nodes)} nodes and {len(new_rels)} rels from KG")
        self.kge.resize(new_nodes, new_rels)

    def _build_entity_ref(
        self, response: str, kb_triples: List[Triple], new_nodes: Set[str], new_rels: Set[str]
    ) -> Tuple[Optional[Dict[str, Tuple[str, str, int, str]]], int]:
        response_ents = [(ent, kg_ent) for ent, kg_ent, _ in self.ner.extract(response)]
        if not response_ents:
            return None, 0

        entity_refs = OrderedDict()
        unmatched_triples = set()
        seen_kg_ents = set()
        for ent, kg_ent in response_ents:
            if kg_ent in seen_kg_ents:
                continue
            seen_kg_ents.add(kg_ent)
            for t in range(len(kb_triples) - 1, -1, -1):
                triple = kb_triples[t]
                if ent.lower() == triple.subject.lower():
                    if kg_ent in self.ner.knowledge_graph[triple.object]:
                        entity_refs[ent] = (triple.object, triple.predicate, 0, kg_ent)
                        break
                    else:
                        unmatched_triples.add((triple.subject, triple.predicate, triple.object))
                elif ent.lower() == triple.object.lower():
                    if kg_ent in self.ner.knowledge_graph[triple.subject]:
                        entity_refs[ent] = (triple.subject, triple.predicate, 1, kg_ent)
                        break
                    else:
                        unmatched_triples.add((triple.subject, triple.predicate, triple.object))

        num_notfound_triples = len(unmatched_triples)

        if not entity_refs:
            return None, num_notfound_triples

        for ent, (pivot, *_) in entity_refs.items():
            for nbr, edges in self.ner.knowledge_graph[pivot].items():
                if not self.kge.contains_node(nbr):
                    new_nodes.add(nbr)
                for rel in edges.keys():
                    if not self.kge.contains_rel(rel):
                        new_rels.add(rel)

        return entity_refs, num_notfound_triples

    def mask_entity(self, tokens: TokenizedText, entity: TokenizedText) -> Tuple[TokenizedText, List[int]]:
        assert len(entity) > 0

        masked_tokens = list(tokens)
        entity_mask = [0] * len(tokens)
        i = 0
        while i < (len(tokens) - len(entity) + 1):
            if entity == tokens[i : i + len(entity)]:
                for j in range(i, i + len(entity)):
                    masked_tokens[j] = self.mlm_tokenizer.mask_token
                    entity_mask[j] = 1
                i += len(entity)
            else:
                i += 1
        return masked_tokens, entity_mask

    def mask_and_collapse_entity(
        self, tokens: TokenizedText, entity: TokenizedText, pad_indices: List[int]
    ) -> Tuple[TokenizedText, List[int]]:
        assert len(entity) > 0

        masked_sequence = []
        i = 0
        pad_index = -1
        while i < len(tokens):
            if entity == tokens[i : i + len(entity)]:
                masked_sequence.append(self.tokenizer.pad_token)
                if pad_index < 0:
                    pad_index = i
                    pad_indices.append(pad_index)
                pad_indices = [pi if pi <= i or pi < 0 else (pi - len(entity) + 1) for pi in pad_indices]
                i += len(entity)
            else:
                masked_sequence.append(tokens[i])
                i += 1

        if pad_index < 0:
            pad_indices.append(pad_index)

        return masked_sequence, pad_indices

    def __getitem__(self, index: int) -> CombinedExample:
        example = self.data[index]

        response = self._word_tokenize(" " + example.response)
        history = [self._word_tokenize(u) for u in example.history]

        if self.include_render and example.render_kb is not None:
            render = self._word_tokenize(" " + example.render_kb)
        else:
            render = None

        lm_inputs, mask_refine_example = None, None
        if self.do_lm:
            lm_inputs = self._build_dialogue_lm(history, render, response, example.kb_triples, example.speaker)

        if example.entities:
            try:
                neighbors, rels, pivots, pivot_fields, labels, label_indices = self._build_from_graph(example.entities)
                # print("**** Neighbors ****: {}".format(neighbors))
                # print("**** Relations ****: {}".format(rels))
                # print("**** Pivots ****: {}".format(pivots))
                # print("**** Labels ****: {}".format(labels))
                # print("**** Labels Indices ****: {}".format(label_indices))
                ordered_entities = sorted(
                    example.entities.keys(), key=lambda x: (len(x.split()), len(x)), reverse=True
                )
                entities = list(example.entities.keys())
                ordered_mask = [entities.index(ent) for ent in ordered_entities]

                if self.mlm_tokenizer is not None:
                    mlm_example = self._build_mlm_from_segments(
                        render, history, example.kb_triples, response, ordered_entities, ordered_mask
                    )
                else:
                    mlm_example = None

                lm_example = self._build_lm_from_segments(
                    render, history, example.kb_triples, response, ordered_entities
                )
            except ValueError as e:
                logger.error(
                    f"Error occurred at index {index} -> "
                    f"response: `{example.response}` - tokenized response: `{response}` - "
                    f"triples: {example.kb_triples} - entities: {example.entities}"
                )
                raise e

            if mlm_example is not None:
                assert len(lm_example.attention_masks) == max(mlm_example.response_entity_mask), (
                    f"At index {index}: #ents: {len(example.entities)} but "
                    f"#lm ents '{len(lm_example.attention_masks)}' fails to match "
                    f"#mlm ents '{max(mlm_example.response_entity_mask)}' -> "
                    f"response: `{example.response}` - tokenized response: `{response}` - "
                    f"triples: {example.kb_triples} - entities: {example.entities}"
                )

            mask_refine_example = MaskRefineExample(
                mlm_example.inputs if mlm_example is not None else labels,
                mlm_example.entity_mask if mlm_example is not None else None,
                lm_example.inputs,
                lm_example.attention_masks,
                neighbors,
                rels,
                pivots,
                pivot_fields,
                labels,
                label_indices,
            )

        return CombinedExample(lm_inputs, mask_refine_example)

    def _build_dialogue_lm(
        self, history: List[str], render: Optional[str], response: str, kb_triples: List[Triple], speaker: int
    ) -> Dict[str, List[Any]]:
        encoded_history = [self.tokenizer.encode(h, add_special_tokens=False) for h in history]
        encoded_response = self.tokenizer.encode(response, add_special_tokens=False)

        triples = []
        if not self.exclude_kb:
            for triple in kb_triples:
                encoded_triple = (
                    self.tokenizer.encode(triple.subject, add_special_tokens=False),
                    self.tokenizer.encode(triple.predicate, add_special_tokens=False),
                    self.tokenizer.encode(triple.object, add_special_tokens=False),
                )

                triples.append(Triple(*encoded_triple))

        if self.include_render and render is not None:
            encoded_render = self.tokenizer.encode(render, add_special_tokens=False)
        else:
            encoded_render = None

        return self._build_from_segments(
            triples,
            encoded_render,
            encoded_history,
            speaker,
            encoded_response,
            with_eos=True,
        )

    def _build_from_graph(
        self, entities: Dict[str, Tuple[str, str, int, str]]
    ) -> Tuple[List[List[int]], List[List[int]], List[int], List[int], List[int], List[int]]:
        neighbors = []
        rels = []
        pivots = []
        pivot_fields = []
        labels = []
        label_indices = []

        for ent, (pivot_ent, pivot_rel, pivot_field, kg_ent) in entities.items():
            pivot_id = self.kge.encode_node(pivot_ent)
            pivots.append(pivot_id)
            pivot_fields.append(pivot_field)

            ent_neighbors = {}
            neighbors_real_int = {}
            ent_id = self.kge.encode_node(kg_ent)
            for object, edges in self.ner.knowledge_graph[pivot_ent].items():
                object_id = self.kge.encode_node(object)

                for rel in edges.keys():
                    if object == kg_ent and rel != pivot_rel:
                        continue

                    # if rel == pivot_rel:
                    # breakpoint()
                    # continue
                    neighbors_real_int[object] = rel
                    rel_id = self.kge.encode_rel(rel)
                    ent_neighbors[object_id] = rel_id

                if object == kg_ent and ent_id not in ent_neighbors:
                    rel = next(iter(edges.keys()))
                    logger.warning(
                        f"`{pivot_rel}` not found and replaced with `{rel}` for `{object}`, neighbor of `{pivot_ent}`"
                    )

                    rel_id = self.kge.encode_rel(rel)
                    ent_neighbors[object_id] = rel_id
                    neighbors_real_int[object] = rel

            if ent_id not in ent_neighbors:
                raise ValueError(
                    f"`{kg_ent}` ({ent_id}) appeared as `{ent}` not found as a neighbor of `{pivot_ent}` "
                    f"with relation with relation `{pivot_rel}`: {ent_neighbors}"
                )

            nbr_list = [nbr for nbr, _ in ent_neighbors.items()]
            if self.max_adjacents > 0 and len(nbr_list) > self.max_adjacents:
                if self.max_adjacents > 1:
                    nbr_list.remove(ent_id)
                    nbr_list = np.random.choice(nbr_list, size=self.max_adjacents - 1, replace=False).tolist()
                    nbr_list.insert(np.random.randint(len(nbr_list)), ent_id)
                else:
                    nbr_list = [ent_id]

            neighbors.append(nbr_list)
            label_indices.append(nbr_list.index(ent_id))
            labels.append(ent_id)

            rels.append([ent_neighbors[nbr] for nbr in nbr_list])
        return neighbors, rels, pivots, pivot_fields, labels, label_indices

    def _build_mlm_from_segments(
        self,
        render: Optional[str],
        history: List[str],
        triples: List[Triple],
        response: str,
        entities: Iterable[str],
        ordered_mask: List[int],
    ) -> MLMExample:
        mlm_tokens = [self.mlm_tokenizer.bos_token_id]
        for utterance in history:
            mlm_tokens.extend(self.mlm_tokenizer.encode(utterance, add_special_tokens=False))

        mlm_tokens.append(self.mlm_tokenizer.sep_token_id)

        if not self.exclude_kb:
            for triple in triples:
                mlm_tokens.extend(triple.encode(self.mlm_tokenizer))
                mlm_tokens.append(self.mlm_tokenizer.sep_token_id)

        if self.include_render and render is not None:
            mlm_tokens.extend(self.mlm_tokenizer.encode(render, add_special_tokens=False))
            mlm_tokens.append(self.mlm_tokenizer.sep_token_id)

        mlm_response = self.mlm_tokenizer.tokenize(response)
        response_entity_mask = [0] * len(mlm_response)

        for i, ent in enumerate(entities):
            mlm_tokenized_ent = self.mlm_tokenizer.tokenize(" " + ent)
            if not mlm_tokenized_ent:
                raise ValueError(f"Entity `{ent}` tokenized to empty!")

            mlm_response, mlm_curr_entity_mask = self.mask_entity(mlm_response, mlm_tokenized_ent)
            response_entity_mask = [
                m or ((ordered_mask[i] + 1) if n else 0) for m, n in zip(response_entity_mask, mlm_curr_entity_mask)
            ]

        if all(m == 0 for m in response_entity_mask):
            raise ValueError("Entity mask must not contain all zeros")

        if len(response_entity_mask) != len(mlm_response):
            raise ValueError("Entity mask and response do not have the same length")

        entity_mask = [0] * len(mlm_tokens) + response_entity_mask
        mlm_tokens.extend(self.mlm_tokenizer.convert_tokens_to_ids(mlm_response))

        return MLMExample(
            inputs=dict(
                input_ids=mlm_tokens,
                attention_mask=[1] * len(mlm_tokens),
            ),
            response_entity_mask=response_entity_mask,
            entity_mask=entity_mask,
        )

    def _build_lm_from_segments(
        self, render: Optional[str], history: List[str], triples: List[Triple], response: str, entities: Iterable[str]
    ) -> LMExample:
        lm_tokens = [self.tokenizer.bos_token_id]
        for utterance in history:
            lm_tokens.extend(self.tokenizer.encode(utterance, add_special_tokens=False))

        lm_tokens.append(self.tokenizer.sep_token_id)

        if not self.exclude_kb:
            for triple in triples:
                lm_tokens.extend(triple.encode(self.tokenizer))
                lm_tokens.append(self.tokenizer.sep_token_id)

        if self.include_render and render is not None:
            lm_tokens.extend(self.tokenizer.encode(render, add_special_tokens=False))
            lm_tokens.append(self.tokenizer.sep_token_id)

        lm_response = self.tokenizer.tokenize(response)

        pad_indices = []
        for i, ent in enumerate(entities):
            lm_tokenized_ent = self.tokenizer.tokenize(" " + ent)
            if not lm_tokenized_ent:
                raise ValueError(f"Entity `{ent}` tokenized to empty!")

            lm_response, pad_indices = self.mask_and_collapse_entity(lm_response, lm_tokenized_ent, pad_indices)

        lm_attention_masks = []
        for pad_index in pad_indices:
            if pad_index >= 0:
                attn_mask = [1] * (pad_index + 1) + [0] * (len(lm_response) - pad_index - 1)
            else:
                attn_mask = [0] * len(lm_response)

            lm_attention_masks.append([1] * len(lm_tokens) + attn_mask)

        lm_tokens.extend(self.tokenizer.convert_tokens_to_ids(lm_response))

        return LMExample(dict(input_ids=lm_tokens), lm_attention_masks)


class GenPipelineDataset(ConversationalDataset):
    def __init__(
        self,
        dataset_path: str,
        lm_tokenizer: PreTrainedTokenizerBase,
        halluc_tokenizer: Optional[PreTrainedTokenizerBase] = None,
        max_history: int = 1,
        exclude_kb: bool = False,
        refine_only: bool = False,
        include_render: bool = False,
    ):
        self.halluc_tokenizer = halluc_tokenizer
        self.refine_only = refine_only

        if self.refine_only:
            assert self.halluc_tokenizer is not None, "when `refine_only` is set, halluc_tokenizer should be set"

        if self.halluc_tokenizer is not None:
            self.halluc_special_tokens = SpecialTokens(*self.halluc_tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS))
        else:
            self.halluc_special_tokens = None

        super().__init__(
            dataset_path,
            lm_tokenizer,
            max_history=max_history,
            max_seq_length=0,
            exclude_kb=exclude_kb,
            spacy_tokenize=True,
            include_render=include_render,
            is_generation=True,
        )

    def _flatten(self, data: List[Dict[str, Any]]) -> Iterable[Union[Example, DialogueInferenceExample]]:
        if not self.refine_only:
            yield from super()._flatten(data)
        else:
            num_no_triple = 0
            num_incomplete_triple = 0

            for dialogue in data:
                history = dialogue["history"][-self.max_history :]
                speaker = dialogue["speaker"]
                if speaker.lower() in ("user", "wizard"):
                    speaker = self.special_tokens.speaker2
                else:
                    speaker = self.special_tokens.speaker1

                if self.max_history > 0:
                    history = history[-self.max_history :]
                response = dialogue["response"]
                if dialogue["knowledge_base"]:
                    if "paths" not in dialogue["knowledge_base"]:
                        num_incomplete_triple += 1
                    else:
                        kb_triples = list(Triple.of(dialogue["knowledge_base"]["paths"]))
                        if len(kb_triples) < len(dialogue["knowledge_base"]["paths"]):
                            num_incomplete_triple += 1
                    render_kb = dialogue["knowledge_base"]["render"]
                else:
                    num_no_triple += 1
                    kb_triples = []
                    render_kb = None
                dialogue_id = dialogue["dialogue_id"]

                generated_responses = dialogue["generated_response"]

                yield DialogueInferenceExample(
                    dlg_id=dialogue_id,
                    history=history,
                    speaker=speaker,
                    response=generated_responses,
                    original_response=response,
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

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if self.refine_only:
            item_dict = {}
        else:
            item_dict = super().__getitem__(index)

        if self.halluc_tokenizer is not None:
            example = self.data[index]

            tokens = [self.halluc_tokenizer.cls_token_id]
            for h in example.history:
                tokens.extend(self.halluc_tokenizer.encode(h, add_special_tokens=False))

            if example.kb_triples:
                for triple in example.kb_triples:
                    tokens.extend(triple.encode(self.halluc_tokenizer))
                    tokens.append(self.halluc_tokenizer.sep_token_id)
            else:
                tokens.append(self.halluc_tokenizer.sep_token_id)

            if self.include_render and example.render_kb is not None:
                tokens.extend(self.halluc_tokenizer.encode(example.render_kb, add_special_tokens=False))
                tokens.append(self.halluc_tokenizer.sep_token_id)

            if self.refine_only:
                halluc_tokens = []
                for generated_response in example.response:
                    resp = " " + self._word_tokenize(generated_response)
                    encoded_resp = self.halluc_tokenizer.encode(resp, add_special_tokens=False)
                    halluc_tokens.append(tokens + encoded_resp + [self.halluc_tokenizer.eos_token_id])

                item_dict["halluc_input_lengths"] = len(tokens)
                tokens = halluc_tokens

            item_dict["halluc_input_ids"] = tokens

        return item_dict


class MaskRefineInferenceDataset(MaskRefineDataset):
    def __init__(
        self,
        dataset_path: str,
        lm_tokenizer: PreTrainedTokenizerBase,
        mlm_tokenizer: Optional[PreTrainedTokenizerBase],
        ner: NER,
        kge: KnowledgeGraphEmbedding,
        max_adjacents: int = 0,
        max_history: int = 1,
        max_seq_length: int = 0,
        exclude_kb: bool = False,
        pivot_field: str = "subject+object",
        include_render: bool = False,
        use_ents_as_hallucination: bool = False,
    ):
        self.pivot_field = pivot_field
        self.use_ents_as_hallucination = use_ents_as_hallucination

        super().__init__(
            dataset_path,
            lm_tokenizer,
            mlm_tokenizer,
            ner,
            kge,
            max_adjacents,
            max_history,
            max_seq_length,
            exclude_kb,
            include_render,
            do_lm=False,
        )

    def _flatten(self, data: List[Dict[str, Any]]) -> Iterable[DialogueEntityExample]:
        num_pivots_notfound = 0
        num_no_kbs = 0
        num_incomplete_kbs = 0
        num_ignored_halluc_preds = 0
        total_halluc_preds = 0
        for dialogue in tqdm(data, desc="Reading data for mask refine"):
            if not dialogue["knowledge_base"] or not dialogue["knowledge_base"].get("paths", None):
                num_no_kbs += 1
                continue

            kb_triples = list(Triple.of(dialogue["knowledge_base"]["paths"]))
            if not kb_triples:
                num_incomplete_kbs += 1
                continue

            speaker = dialogue["speaker"]
            if speaker.lower() in ("user", "wizard"):
                speaker = self.special_tokens.speaker2
            else:
                speaker = self.special_tokens.speaker1

            response = _normalize(dialogue["response"])
            render_kb = dialogue["knowledge_base"]["render"]
            dialogue_id = dialogue["dialogue_id"]

            history = dialogue["history"]
            if self.max_history > 0:
                history = history[-self.max_history :]

            ref_triple = kb_triples[0]
            pivots = {}
            if self.pivot_field in ("subject", "subject+object"):
                if self.kge.contains_node(ref_triple.subject):
                    pivots["subject"] = ref_triple.subject
            if self.pivot_field in ("object", "subject+object"):
                if self.kge.contains_node(ref_triple.object):
                    pivots["object"] = ref_triple.object

            if not pivots:
                num_pivots_notfound += 1
                continue

            hallucination_predictions = dialogue.get("hallucination_preds", None)

            if not hallucination_predictions or self.use_ents_as_hallucination:
                hallucination_predictions = []
                for generated_response in dialogue["generated_response"]:
                    hallucination_predictions.append([ent for ent, _, _ in self.ner.extract(generated_response)])

            for j, (generated_response, halluc_preds) in enumerate(
                zip(dialogue["generated_response"], hallucination_predictions)
            ):
                if not halluc_preds:
                    continue

                generated_response = _normalize(generated_response)

                response_tokens = self._word_tokenize(generated_response, as_string=False)
                acceptable_preds = []
                for halluc_ent in halluc_preds:
                    ent_tokens = self._word_tokenize(halluc_ent, as_string=False)
                    for t in range(len(response_tokens)):
                        if ent_tokens == response_tokens[t : t + len(ent_tokens)]:
                            acceptable_preds.append(halluc_ent)
                            break

                num_ignored_halluc_preds += len(halluc_preds) - len(acceptable_preds)
                total_halluc_preds += len(halluc_preds)

                if not acceptable_preds:
                    continue

                if self.max_seq_length > 0:
                    encoded_len = (
                        sum(len(t.encode(self.tokenizer)) + 1 for t in kb_triples)
                        + len(self.tokenizer.tokenize(" ".join(history) + " " + generated_response))
                        + len(history)  # speakers
                        + 1  # BOS
                        + 1  # EOS
                    )
                    if encoded_len >= self.max_seq_length:
                        continue

                entity_refs = OrderedDict()
                for halluc_ent in halluc_preds:
                    if not halluc_ent:
                        continue
                    entity_refs[halluc_ent] = (pivots, ref_triple.predicate)

                if not entity_refs:
                    continue

                yield DialogueInferenceExample(
                    dlg_id=dialogue_id,
                    history=history,
                    speaker=speaker,
                    response=generated_response,
                    kb_triples=kb_triples,
                    render_kb=render_kb,
                    original_response=response,
                    response_idx=j,
                    entities=entity_refs,
                )

        logger.info(f"#instances with no KB: {num_no_kbs} ({100. * num_no_kbs / len(data):.2f}%)")
        logger.info(
            f"#instances with incomplete KB: {num_incomplete_kbs} ({100. * num_incomplete_kbs / len(data):.2f}%)"
        )
        logger.info(
            f"#instances with not found pivots: {num_pivots_notfound} ({100. * num_pivots_notfound / len(data):.2f}%)"
        )
        logger.info(
            f"#ignored hallucination preds: {num_ignored_halluc_preds} out of {total_halluc_preds} "
            f"({100. * num_ignored_halluc_preds / total_halluc_preds:.2f}%)"
        )

    def _build_from_graph(
        self, entities: Dict[str, Tuple[Dict[str, str], str]]
    ) -> Tuple[List[Any], List[Any], List[List[int]], List[List[int]], Optional[List[int]], Optional[List[int]]]:
        candidates = []
        candidate_rels = []
        all_pivots = []
        all_pivot_fields = []

        for ent, (pivot_ents, pivot_rel) in entities.items():
            neighbors, rels = [], []
            pivots = []
            pivot_fields = []
            for pivot_field, pivot_ent in pivot_ents.items():
                pivot_id = self.kge.encode_node(pivot_ent)
                pivots.append(pivot_id)
                pivot_fields.append(0 if pivot_field == "subject" else 1)

                ent_neighbors = {}
                for object, edges in self.ner.knowledge_graph[pivot_ent].items():
                    if not self.kge.contains_node(object):
                        continue

                    object_id = self.kge.encode_node(object)

                    if pivot_rel in edges:
                        rel = pivot_rel
                    else:
                        rel = next(iter(edges.keys()))

                    if not self.kge.contains_rel(rel):
                        for e in takewhile(lambda x: True, iter(edges.keys())):
                            if self.kge.contains_rel(e):
                                rel = e
                                break
                            else:
                                rel = None

                        if not rel:
                            continue

                    rel_id = self.kge.encode_rel(rel)
                    ent_neighbors[object_id] = rel_id

                nbr_list = [nbr for nbr, _ in ent_neighbors.items()]
                if self.max_adjacents > 0 and len(nbr_list) > self.max_adjacents:
                    nbr_list = np.random.choice(nbr_list, size=self.max_adjacents, replace=False).tolist()

                neighbors.append(nbr_list)
                rels.append([ent_neighbors[nbr] for nbr in nbr_list])
            all_pivots.append(pivots)
            all_pivot_fields.append(pivot_fields)
            candidates.append(neighbors)
            candidate_rels.append(rels)

        return candidates, candidate_rels, all_pivots, all_pivot_fields, None, None


class MaskRerankInferenceDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        tokenizer: PreTrainedTokenizerBase,
        rerank_size: int,
        is_encoder_decoder: bool,
        max_history: int = 1,
        max_seq_length: int = 0,
        exclude_kb: bool = False,
        include_render: bool = False,
    ):
        self.tokenizer = tokenizer
        self.rerank_size = rerank_size
        self.is_encoder_decoder = is_encoder_decoder
        self.max_history = max_history
        self.max_seq_length = max_seq_length
        self.exclude_kb = exclude_kb
        self.include_render = include_render

        self.nlp = spacy.load("en_core_web_sm", disable=("ner", "parser", "tagger", "lemmatizer"))

        with open(dataset_path, "r", encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f]
            self.data = [d for d in self.data if d["refines"]]

        logger.info(f"#data loaded: {len(self.data)}")

    def _word_tokenize(self, text: str) -> str:
        # To resolve issues like 'He also directed Batman R.I.P.. Have you seen that one?'
        text = re.sub(r"(\w+\.)\.\s", r"\1 . ", text)
        text = re.sub(r"(\.\w\.)\.\s", r"\1 . ", text)

        # To resolve issues like 'I like Neil Brown Jr..' and 'I think he plays for Real Madrid C.F..'
        if re.match(r".*\w+\.\.$", text) or re.match(r".*\.\w\.\.$", text):
            text = text[:-1] + " ."

        tokens = [tok.text for tok in self.nlp(text)]
        return " ".join(tokens)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self.data[index]

        source_tokens = []

        if not self.is_encoder_decoder:
            source_tokens.append(self.tokenizer.bos_token_id)

        for h in example["history"]:
            source_tokens.extend(self.tokenizer.encode(self._word_tokenize(h)))

        if self.is_encoder_decoder:
            source_tokens.append(self.tokenizer.eos_token_id)

        if not self.exclude_kb:
            triples = Triple.of(example["knowledge_base"]["paths"])
            for triple in triples:
                source_tokens.extend(triple.encode(self.tokenizer, add_sep=False))
                if self.is_encoder_decoder:
                    source_tokens.append(self.tokenizer.eos_token_id)

        if self.include_render and example.render_kb is not None:
            source_tokens.extend(self.tokenizer.encode(self._word_tokenize(example.render_kb)))
            if self.is_encoder_decoder:
                source_tokens.append(self.tokenizer.eos_token_id)

        source_ids = []
        source_mask = []
        target_ids = []
        target_mask = []
        replace_masks = []

        for refine in example["refines"]:
            refined_responses = refine["refined_responses"][: self.rerank_size]
            replacements = refine["replacements"][: self.rerank_size]

            refine_source_ids = [list(source_tokens) for _ in range(len(refined_responses))]
            refine_source_mask = [[1] * len(source_tokens) for _ in range(len(refined_responses))]
            refine_target_ids = []
            refine_target_mask = []
            refine_replace_mask = []

            for refined_resp, repl in zip(refined_responses, replacements):
                repl_token_ids = self.tokenizer.encode(" " + self._word_tokenize(repl), add_special_tokens=False)
                resp_token_ids = self.tokenizer.encode(" " + self._word_tokenize(refined_resp))
                resp_token_ids.append(self.tokenizer.eos_token_id)

                replace_mask = [0] * len(resp_token_ids)
                for j in range(len(resp_token_ids) - len(repl_token_ids)):
                    if resp_token_ids[j : j + len(repl_token_ids)] == repl_token_ids:
                        for i in range(j, j + len(repl_token_ids)):
                            replace_mask[i] = 1
                        break

                refine_target_ids.append(resp_token_ids)
                refine_target_mask.append([1] * len(resp_token_ids))
                refine_replace_mask.append(replace_mask)

            if self.is_encoder_decoder:
                source_ids.append(refine_source_ids)
                source_mask.append(refine_source_mask)
                target_ids.append(refine_target_ids)
                target_mask.append(refine_target_mask)
                replace_masks.append(refine_replace_mask)
            else:
                source_ids.append([src + tgt for src, tgt in zip(refine_source_ids, refine_target_ids)])
                source_mask.append([src + tgt for src, tgt in zip(refine_source_mask, refine_target_mask)])
                replace_masks.append(
                    [([0] * len(src)) + repl for src, repl in zip(refine_source_mask, refine_replace_mask)]
                )

        output = dict(
            input_ids=source_ids,
            attention_mask=source_mask,
            replace_mask=replace_masks,
        )

        if self.is_encoder_decoder:
            output["decoder_input_ids"] = target_ids
            output["decoder_attention_mask"] = target_mask

        return output


class MixedDataset(Dataset):
    def __init__(self, conv_dataset: ConversationalDataset, mask_refine_dataset: MaskRefineDataset):
        self.conv_dataset = conv_dataset
        self.mask_refine_dataset = mask_refine_dataset

    def __len__(self) -> int:
        return len(self.conv_dataset) + len(self.mask_refine_dataset)

    def __getitem__(self, index: int) -> Union[MaskRefineExample, Dict[str, Any]]:
        if index < len(self.conv_dataset):
            return self.conv_dataset[index]
        else:
            return self.mask_refine_dataset[index - len(self.conv_dataset)]


@dataclass
class MaskRerankCollator:
    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_ents = max([len(ex["input_ids"]) for ex in examples])
        max_cands = max([len(cands) for ex in examples for cands in ex["input_ids"]])

        batch = {}
        for field in ("input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "replace_mask"):
            if not all(field in ex for ex in examples):
                continue

            max_seq_l = max([len(m) for ex in examples for cands in ex[field] for m in cands])
            if self.pad_to_multiple_of:
                max_seq_l = int(self.pad_to_multiple_of * np.ceil(max_seq_l / self.pad_to_multiple_of))

            pad = 0 if field.endswith("_mask") else self.tokenizer.pad_token_id
            padded_field = np.full((len(examples), max_ents, max_cands, max_seq_l), pad, dtype=np.int)
            for i, ex in enumerate(examples):
                for j, cands in enumerate(ex[field]):
                    for k, m in enumerate(cands):
                        padded_field[i, j, k, : len(m)] = m

            batch[field] = torch.from_numpy(padded_field)

        return batch


@dataclass
class MaskRefineCollator:
    lm_tokenizer: PreTrainedTokenizerBase
    mlm_tokenizer: Optional[PreTrainedTokenizerBase]
    multiple_pivots_allowed: bool = False
    kg_pad: int = 0
    label_pad: int = -100
    padding: str = "longest"
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __post_init__(self):
        self.lm_collator = LMCollator(
            self.lm_tokenizer.pad_token_id,
            self.kg_pad,
            label_pad=self.label_pad,
            as_tuple=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

    def __call__(self, examples: List[CombinedExample]) -> Tuple[Optional[Tuple[Any, ...]], Optional[Tuple[Any, ...]]]:
        """
        features is what we return from __getitem__
        """
        mask_refine_examples = [ex.mask_refine for ex in examples if ex.has_mask_refine]
        lm_examples = [ex.lm_inputs for ex in examples if ex.has_lm]

        nce_batch, lm_batch = None, None
        if mask_refine_examples:
            lm_input_ids = self.lm_tokenizer.pad(
                [ex.inputs for ex in mask_refine_examples],
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )["input_ids"]

            max_ents = max([len(ex.attention_masks) for ex in mask_refine_examples])
            max_seq_l = max([len(m) for ex in mask_refine_examples for m in ex.attention_masks])
            if self.pad_to_multiple_of:
                max_seq_l = int(self.pad_to_multiple_of * np.ceil(max_seq_l / self.pad_to_multiple_of))

            padded_attention_mask = np.zeros((len(mask_refine_examples), max_ents, max_seq_l), dtype=np.int)
            for i, ex in enumerate(mask_refine_examples):
                for j, m in enumerate(ex.attention_masks):
                    padded_attention_mask[i, j, : len(m)] = m

            if self.mlm_tokenizer is not None:
                max_mlm_l = max([len(ex.mlm_entity_mask) for ex in mask_refine_examples])
                if self.pad_to_multiple_of:
                    max_mlm_l = int(self.pad_to_multiple_of * np.ceil(max_mlm_l / self.pad_to_multiple_of))

                padded_mlm_entity_mask = np.zeros((len(mask_refine_examples), max_mlm_l), dtype=np.int)
                for i, ex in enumerate(mask_refine_examples):
                    padded_mlm_entity_mask[i, : len(ex.mlm_entity_mask)] = ex.mlm_entity_mask
            else:
                padded_mlm_entity_mask = None

            contains_labels = all(ex.labels is not None for ex in mask_refine_examples)
            if contains_labels:
                max_label_l = max([len(ex.labels) for ex in mask_refine_examples])
                assert max_label_l == max_ents
            else:
                max_label_l = max([len(ex.candidate_ids) for ex in mask_refine_examples])

            if self.multiple_pivots_allowed:
                max_pivots = max([len(pivots) for ex in mask_refine_examples for pivots in ex.pivot_ids])
                max_cands = max(
                    [
                        len(cands)
                        for ex in mask_refine_examples
                        for pivot_cands in ex.candidate_ids
                        for cands in pivot_cands
                    ]
                )
                padded_candidate_ids = self._pad4D(
                    mask_refine_examples, "candidate_ids", max_label_l, max_pivots, max_cands
                )
                padded_candidate_rels = self._pad4D(
                    mask_refine_examples, "candidate_rels", max_label_l, max_pivots, max_cands
                )
                padded_pivots = self._pad3D(mask_refine_examples, "pivot_ids", max_label_l, max_pivots)
                padded_pivot_fields = self._pad3D(
                    mask_refine_examples, "pivot_fields", max_label_l, max_pivots, pad=-1
                )
            else:
                max_cands = max([len(cands) for ex in mask_refine_examples for cands in ex.candidate_ids])
                padded_candidate_ids = self._pad3D(mask_refine_examples, "candidate_ids", max_label_l, max_cands)
                padded_candidate_rels = self._pad3D(mask_refine_examples, "candidate_rels", max_label_l, max_cands)
                padded_pivots = self._pad2D(mask_refine_examples, "pivot_ids", max_label_l, self.kg_pad)
                padded_pivot_fields = None

            if self.mlm_tokenizer is not None:
                mlm_batch = self.mlm_tokenizer.pad(
                    [ex.mlm_inputs for ex in mask_refine_examples],
                    padding=self.padding,
                    max_length=self.max_length,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                    return_attention_mask=True,
                    return_tensors="pt",
                )
            else:
                mlm_batch = (
                    self._pad2D(mask_refine_examples, "labels", max_label_l, self.kg_pad) if contains_labels else None
                )

            padded_labels = (
                self._pad2D(mask_refine_examples, "labels", max_label_l, self.label_pad) if contains_labels else None
            )

            nce_batch = (
                mlm_batch,
                torch.from_numpy(padded_mlm_entity_mask) if padded_mlm_entity_mask is not None else None,
                lm_input_ids,
                torch.from_numpy(padded_attention_mask),
                padded_candidate_ids,
                padded_candidate_rels,
                padded_pivots,
                padded_pivot_fields,
                padded_labels,
                self._pad2D(mask_refine_examples, "label_indices", max_label_l, self.label_pad)
                if contains_labels
                else None,
            )

        if lm_examples:
            lm_batch = self.lm_collator(lm_examples)

        return (nce_batch, lm_batch)

    def _pad2D(self, examples: List[MaskRefineExample], attr: str, E: int, pad: int):
        padded = np.full((len(examples), E), pad, dtype=np.int)

        for i, ex in enumerate(examples):
            labels = getattr(ex, attr)
            padded[i, : len(labels)] = labels

        return torch.from_numpy(padded)

    def _pad3D(self, examples: List[MaskRefineExample], attr: str, E: int, C: int, pad: int = None):
        padded = np.full((len(examples), E, C), pad or self.kg_pad, dtype=np.int)

        for i, ex in enumerate(examples):
            for j, cands in enumerate(getattr(ex, attr)):
                padded[i, j, : len(cands)] = cands

        return torch.from_numpy(padded)

    def _pad4D(self, examples: List[MaskRefineExample], attr: str, E: int, P: int, C: int):
        padded = np.full((len(examples), E, P, C), self.kg_pad, dtype=np.int)

        for i, ex in enumerate(examples):
            for j, pivot_cands in enumerate(getattr(ex, attr)):
                for k, cands in enumerate(pivot_cands):
                    padded[i, j, k, : len(cands)] = cands

        return torch.from_numpy(padded)


@dataclass
class MixedCollator:
    nce_collator: MaskRefineCollator
    lm_collator: LMCollator

    def __call__(
        self, examples: List[Union[CombinedExample, Dict[str, Any]]]
    ) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
        nce_examples = [ex for ex in examples if isinstance(ex, CombinedExample)]
        lm_examples = [ex for ex in examples if not isinstance(ex, CombinedExample)]

        nce_batch, lm_batch = None, None
        if nce_examples:
            nce_batch = self.nce_collator(nce_examples)[0]

        if lm_examples:
            lm_batch = self.lm_collator(lm_examples)

        return (nce_batch, lm_batch)


@dataclass
class GenPipelineCollator:
    lm_tokenizer: PreTrainedTokenizerBase
    halluc_tokenizer: Optional[PreTrainedTokenizerBase] = None
    kg_pad: int = 0
    padding: str = "longest"
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __post_init__(self):
        self.lm_collator = LMCollator(
            self.lm_tokenizer.pad_token_id,
            self.kg_pad,
            pad_to_multiple_of=self.pad_to_multiple_of,
            fields=(
                "input_ids",
                "token_type_ids",
                "attention_mask",
            ),
        )

    def __call__(self, examples: List[Dict[str, Any]]) -> Tuple[Any, ...]:
        lm_batch = self.lm_collator(examples)

        if self.halluc_tokenizer is not None:
            halluc_inputs = dict(
                input_ids=[ex["halluc_input_ids"] for ex in examples],
            )
            halluc_padded_inputs = self.halluc_tokenizer.pad(
                halluc_inputs,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )["input_ids"]
        else:
            halluc_padded_inputs = None

        return (
            lm_batch["input_ids"],
            lm_batch.get("token_type_ids", None),
            lm_batch["attention_mask"],
            halluc_padded_inputs,
        )


@dataclass
class HallucOnlyCollator:
    tokenizer: PreTrainedTokenizerBase
    kg_pad: int = 0
    padding: str = "longest"
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, examples: List[Dict[str, Any]]) -> Tuple[Any, ...]:
        halluc_inputs = dict(
            input_ids=[gen for ex in examples for gen in ex["halluc_input_ids"]],
        )

        padded_inputs = self.tokenizer.pad(
            halluc_inputs,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_attention_mask=True,
        )

        max_gen_num = max(len(ex["halluc_input_ids"]) for ex in examples)

        reshaped_padded_inputs = np.full(
            (len(examples), max_gen_num, len(padded_inputs["input_ids"][0])), self.tokenizer.pad_token_id
        )

        reshaped_padded_attention_mask = np.zeros((len(examples), max_gen_num, len(padded_inputs["input_ids"][0])))

        k = 0
        for i, ex in enumerate(examples):
            for j, input_ids in enumerate(ex["halluc_input_ids"]):
                reshaped_padded_inputs[i, j] = padded_inputs["input_ids"][k]
                reshaped_padded_attention_mask[i, j] = padded_inputs["attention_mask"][k]
                k += 1

        return (
            dict(
                input_ids=torch.LongTensor(reshaped_padded_inputs),
                attention_mask=torch.LongTensor(reshaped_padded_attention_mask),
            ),
            torch.LongTensor([ex["halluc_input_lengths"] for ex in examples]),
        )
