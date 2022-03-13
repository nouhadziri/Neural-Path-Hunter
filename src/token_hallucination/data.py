import json
import logging
import os
import pickle
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from random import randrange
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import spacy
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from ..conv_data import Triple, EncodedText
from ..graph import EntityTagger, load_kg, refine_node

logger = logging.getLogger("token_hallucination_data")


@dataclass
class NamedEntity:
    text: str
    label: Optional[str] = None
    start_char: Optional[str] = None
    end_char: Optional[str] = None

    def __repr__(self) -> str:
        return f"{self.text}/{self.label}" if self.label else self.text

    def __eq__(self, o) -> bool:
        return self.text == o.text

    def __ne__(self, o: object) -> bool:
        return not (self == o)

    def __hash__(self) -> int:
        return hash(self.text)


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


@dataclass
class Example:
    dlg_id: int
    history: List[str]
    response: str
    kb_triples: List[Triple]
    render_kb: Optional[str]
    corrupted_response: str
    swap_map: Dict[str, str]

    def asdict(self):
        return dict(
            dialogue_id=self.dlg_id,
            history=self.history,
            knowledge_base={
                "paths": [[t.subject, t.predicate, t.object] for t in self.kb_triples],
                "render": self.render_kb,
            },
            response=self.response,
            corrupted_response=self.corrupted_response,
            swaps=[{"old": old, "new": new} for old, new in self.swap_map.items()],
        )

    @classmethod
    def from_dict(cls, item_dict: Dict[str, Any]) -> "Example":
        return Example(
            dlg_id=item_dict["dialogue_id"],
            history=item_dict["history"],
            response=item_dict["response"],
            kb_triples=[Triple(p[0], p[1], p[2]) for p in item_dict["knowledge_base"]["paths"]],
            render_kb=item_dict["knowledge_base"]["render"],
            corrupted_response=item_dict["corrupted_response"],
            swap_map={sw["old"]: sw["new"] for sw in item_dict["swaps"]},
        )


class ResponseCorruptedDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        dataset_path: str,
        output_path: str,
        ent_fn: Callable[[str], List[NamedEntity]],
        graph_file: str,
        output: str,
        num_swaps: int,
        num_corrupted_samples: int,
        max_attempts: int,
        max_history: int = 1,
        max_seq_length: int = 0,
        entity_pickle_path: Optional[str] = None,
        overwrite_corrupted_data: bool = False,
        is_inference: bool = False,
        use_groundtruth_response: bool = False,
        include_response: bool = False,
    ):
        """
        :param num_swaps: Number of entities swap in the history.
        :param num_corrupted_samples: Number of corrupted samples generated based on the history.
        :param max_attempts: Max number of attempts to swap entities.
        """
        self.tokenizer = tokenizer
        self.ent_fn = ent_fn
        # self.no_spacy_entity = no_spacy_entity
        self.graph_file = graph_file
        self.output_path = output
        self.max_history = max_history
        self.num_swaps = num_swaps
        self.num_corrupted_samples = num_corrupted_samples
        self.max_attempts = max_attempts
        self.max_seq_length = max_seq_length
        self.is_inference = is_inference
        self.use_groundtruth_response = use_groundtruth_response
        self.include_response = include_response

        if self.is_inference and self.use_groundtruth_response:
            assert not self.include_response, "include_response should not be set when use_groundtruth_response is true"

        self.max_noncorruptable_samples = 3735
        self.max_augmented_triples = 100
        self._nlp = spacy.load("en_core_web_sm", disable=("ner", "parser", "tagger", "lemmatizer"))
        self.graph = load_kg(self.graph_file)
        self.entity_tagger = EntityTagger(self.graph, output_path)
        prefix = Path(dataset_path).with_suffix("").name

        if not self.is_inference:
            self.triples_dict = self.get_relation_triples(self.graph_file, self.output_path)

            self._ner_data_path = os.path.join(
                output_path, f"{prefix}_swaps{num_swaps}_ncorrupts{num_corrupted_samples}_seq{max_seq_length}.jsonl"
            )
            if not os.path.isfile(self._ner_data_path) or overwrite_corrupted_data:
                logger.info(f"Building dataset and save into `{self._ner_data_path}`")
                self.entities = self.get_dialog_entities(dataset_path, self.output_path, entity_pickle_path)

                with open(dataset_path, "r", encoding="utf-8") as f:
                    data = [json.loads(line) for line in f]

                self.data = list(self._build(data, graph_file, self.output_path, self._ner_data_path))

                if len(self.data) < len(data):
                    logger.warning(
                        f"data size for `{dataset_path}`: "
                        f"{len(data)} -> {len(self.data)} = {len(self.data) - len(data)} examples "
                        f"~{100 * (len(self.data) / len(data)):.1f}%"
                    )
            else:
                self.data = list(self._load(self._ner_data_path))
                logger.info(f"Dataset loaded from `{self._ner_data_path}`. Size = {len(self.data)} examples")
        else:
            self._ner_data_path = dataset_path
            with open(dataset_path, "r", encoding="utf-8") as f:
                data = [json.loads(line) for line in f]
            self.data = list(self._build(data, graph_file, self.output_path))

            if len(self.data) < len(data):
                logger.warning(
                    f"data size for `{dataset_path}`: "
                    f"{len(data)} -> {len(self.data)} = {len(self.data) - len(data)} examples "
                    f"~{100 * (len(self.data) / len(data)):.1f}%"
                )

    def __len__(self) -> int:
        return len(self.data)

    def _get_entities_not_spacy(self, text) -> List[NamedEntity]:
        # start = time.time()
        # breakpoint()
        tagger = self.entity_tagger
        entities_resp = tagger.extract(text)
        if not entities_resp:
            return []
        # end = time.time()
        # print(f"Runtime of the extracting entities from the response {end - start}")
        return [NamedEntity(e.strip()) for e in entities_resp]

    def _tokenize(self, text: str) -> str:
        tokens = [t.text for t in self._nlp(text)]

        # To resolve issues like this: `He\'s a great author and had a couple starring rolls too."The Stand" comes to mind.`
        # A space between period '.' and '"' will be added
        processed_tokens = []
        for tok in tokens:
            if re.match(r"^\W\w+", tok):
                processed_tokens.append(tok[0])
                processed_tokens.append(tok[1:])
            else:
                processed_tokens.append(tok)

        return " ".join(processed_tokens)

    def get_dialog_entities(self, input_file, output_dir, pickle_file=None):
        if pickle_file is not None and os.path.isfile(pickle_file):
            output_path = Path(pickle_file)
        else:
            prefix = Path(input_file).with_suffix("").name
            output_path = Path(output_dir) / f"{prefix}_entity_types.pkl"

        if output_path.exists():
            with output_path.open("rb") as f:
                l_entities = pickle.load(f)
        else:
            logger.debug("Building entities and their corresponding types")

            with open(input_file, "r") as f:
                entities = defaultdict(set)
                l_entities = defaultdict(list)
                for line in tqdm(f.readlines(), desc="map entities"):
                    data = json.loads(line)
                    if not data["knowledge_base"]:
                        continue
                    history = data["history"]
                    response = data["response"]
                    context = [response] + history
                    for utterance in context:
                        for e in self.ent_fn(utterance):
                            entities[e.label].add(e.text)

            for k, v in entities.items():
                for j in v:
                    l_entities[k].append(j)
            with output_path.open("wb") as out_file:
                pickle.dump(l_entities, out_file)

        return l_entities

    # Build
    def _build(
        self, data: List[Dict[str, Any]], graph_file, output_path, save_path: Optional[str] = None
    ) -> Iterable[Example]:
        """
        Flattens dialogues so that each predictable turn becomes a top level entry.
        """
        if save_path:
            save_file = open(save_path, "w")
        else:
            save_file = None
        num_noncorruptable_samples = 0

        try:
            for dialogue in tqdm(data, total=len(data), desc="read examples"):
                if not dialogue["knowledge_base"] or not dialogue["knowledge_base"].get("paths", None):
                    continue

                history = dialogue["history"]
                if self.max_history > 0:
                    history = history[-self.max_history :]
                response = self._tokenize(" ".join(dialogue["response"].split()))
                kb_gold_triples = [Triple(*triple) for triple in dialogue["knowledge_base"]["paths"]]

                aug_kb_triples = kb_gold_triples
                kb_triples = kb_gold_triples
                render_kb = None
                dialogue_id = dialogue.get("dialogue_id", 0)

                if self.is_inference:
                    if self.use_groundtruth_response:
                        corrupted_responses = [(response, {})]
                    else:
                        generated_response = dialogue["generated_response"]
                        if isinstance(generated_response, (list, tuple)):
                            generated_response = generated_response[0]

                        corrupted_responses = [(self._tokenize(generated_response), {})]
                else:

                    corrupted_responses = self.corrupt_non_spacy_response(response, aug_kb_triples)
                    if corrupted_responses == [response]:
                        num_noncorruptable_samples += 1
                        corrupted_responses = self.corrupt_gt_response(response, kb_triples)

                        if not corrupted_responses:
                            continue

                for corrupted_response, swap_map in corrupted_responses:
                    response_seq = ((response + " ") if self.include_response else "") + corrupted_response
                    if self.max_seq_length > 0:
                        encoded_len = (
                            len(kb_triples) * 6
                            + len(self.tokenizer.tokenize(" ".join(history) + " " + response_seq))
                            + len(history)
                            + 1
                            + 2
                        )
                        if encoded_len >= self.max_seq_length:
                            continue

                    corrupted_example = Example(
                        dlg_id=dialogue_id,
                        history=history,
                        response=response,
                        kb_triples=kb_triples,
                        render_kb=render_kb,
                        corrupted_response=corrupted_response,
                        swap_map=swap_map,
                    )

                    if save_file:
                        save_file.write(json.dumps(corrupted_example.asdict()) + "\n")

                    yield corrupted_example

        finally:
            if save_file:
                save_file.close()

    def _load(self, ner_data_path: str) -> Iterable[Example]:
        with open(ner_data_path, "r") as ner_file:
            for line in ner_file:
                yield Example.from_dict(json.loads(line.strip()))

    def get_relation_triples(self, graph_file, output_dir, pickle_file=None):
        if pickle_file is not None and os.path.isfile(pickle_file):
            output_path = Path(pickle_file)
        else:
            prefix = Path(graph_file).with_suffix("").name
            output_path = Path(output_dir) / f"{prefix}_relation_dict.pkl"

        if output_path.exists():
            with output_path.open("rb") as f:
                relations = pickle.load(f)
        else:
            logger.debug("Retrieving triples based on relations ....")

            with open(graph_file, "r") as f:
                relations = defaultdict(lambda: {"subject": set(), "object": set()})
                for line in tqdm(f.readlines(), desc="Get relations dict"):
                    if len(line.strip().split("\t")) < 3:
                        continue

                    head, edge, tail = line.strip().split("\t")
                    head = refine_node(head)
                    tail = refine_node(tail)
                    if edge.startswith("~"):
                        edge = edge[1:]
                        src, dest = tail, head
                    else:
                        src, dest = head, tail
                    relations[edge]["subject"].add(head)
                    relations[edge]["object"].add(tail)

            with output_path.open("wb") as out_file:
                pickle.dump(dict(relations), out_file)

        return relations

    def corrupt_non_spacy_response(self, response, gt_triples):
        triples_dict = self.triples_dict
        entities = [str(ent) for ent in self._get_entities_not_spacy(response)]

        corrupted_samples = []
        swap_dict = {}
        perturbed_resp = None

        if entities:
            for entity in entities:
                entity = str(entity)
                l_entity = entity.lower()
                for triple in gt_triples:
                    gold_subject, gold_rel, gold_object = (
                        triple.subject.lower(),
                        triple.predicate.lower(),
                        triple.object.lower(),
                    )

                    if gold_rel in triples_dict:
                        retrieved_sub_obj = triples_dict[gold_rel]
                    else:
                        continue
                    if l_entity in gold_subject:
                        selected_index = randrange(len(retrieved_sub_obj["subject"]))

                        retrieved_ent = list(retrieved_sub_obj["subject"])

                        while retrieved_ent[selected_index] == gold_subject:
                            selected_index = randrange(len(retrieved_sub_obj["subject"]))
                        corrupted_ent = retrieved_ent[selected_index]
                        for t in range(self.max_attempts):
                            if is_valid_swap(corrupted_ent, entities):
                                swap_dict[entity] = corrupted_ent
                                break
                            else:
                                selected_index = randrange(len(retrieved_sub_obj["subject"]))
                                corrupted_ent = retrieved_ent[selected_index]

                    elif l_entity in gold_object:
                        selected_index = randrange(len(retrieved_sub_obj["object"]))
                        retrieved_ent = list(retrieved_sub_obj["object"])
                        while retrieved_ent[selected_index] == gold_object:
                            selected_index = randrange(len(retrieved_sub_obj["object"]))

                        corrupted_ent = retrieved_ent[selected_index]
                        for t in range(self.max_attempts):
                            if is_valid_swap(corrupted_ent, entities):
                                swap_dict[entity] = corrupted_ent
                                break
                            else:
                                selected_index = randrange(len(retrieved_sub_obj["object"]))
                                corrupted_ent = retrieved_ent[selected_index]

            if swap_dict:
                perturbed_resp = text_swapping(response, swap_dict)
                corrupted_samples.append((perturbed_resp, swap_dict))
                return corrupted_samples

            if perturbed_resp is None:
                return [response]

        else:
            return [response]

    def corrupt_gt_response(
        self,
        response: str,
        gt_triples: List[Triple],
    ) -> List[Tuple[str, Dict[str, str]]]:
        """
         Swaps entities in the gold response with entities of the same type that are not
         in the vicinity of the starting node.

        :param response: The sentence in which to swap entities.
        :param gt_triples: The KB triples associated with the gold response.
        :param entities: All entities extracted from all the dialogues associated with their types.
                          key = type of entity and value = list of entities

        :return: Tuple of perturbed sentence and number of perturbed examples.
        """
        ent_dict = {ent.text: ent.label for ent in self.ent_fn(response) if ent.label in self.entities}
        # filter entities that are fully subsumed by other entities
        ents = list(e for e in ent_dict.keys() if all(e not in e2 for e2 in ent_dict.keys() if e != e2))

        not_allowed_entities = set(e for t in gt_triples for e in (t.subject, t.object))
        for e in ents:
            not_allowed_entities.add(e)

        corrupted_samples = []
        for n in range(self.num_corrupted_samples):
            if self.num_swaps <= len(ents):
                selected_indices = random.sample(range(len(ents)), k=self.num_swaps)
            else:
                selected_indices = range(len(ents))

            selected_ents = [ents[i].lower() for i in selected_indices]

            swap_dict = {}
            for i in selected_indices:
                selected_ent = ents[i]
                selected_type = ent_dict[selected_ent]
                adversarial_cands = [ent for ent in self.entities[selected_type] if ent not in not_allowed_entities]

                if not adversarial_cands:
                    logger.debug(
                        f"No candidates found for type `{selected_type}` for corrupting `{selected_ent}` in `{response}`"
                    )
                    continue

                for t in range(self.max_attempts):
                    selected_cand_idx = random.randrange(0, len(adversarial_cands))
                    if is_valid_swap(adversarial_cands[selected_cand_idx], selected_ents):
                        swap_dict[selected_ent] = self._tokenize(adversarial_cands[selected_cand_idx])
                        break

            if swap_dict:
                perturbed_resp = text_swapping(response, swap_dict)
                corrupted_samples.append((perturbed_resp, swap_dict))


        return corrupted_samples

    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self.data[index]
        history = example.history
        kb_triples = example.kb_triples
        response = example.response
        corrupted_response = example.corrupted_response

        if self.include_response:
            encoded_response = self.tokenizer.encode(response, add_special_tokens=False)
            history.append(encoded_response)

        encoded_history = []
        for h in history:
            h_ids = self.tokenizer.encode(h, add_special_tokens=False)
            encoded_history.append(h_ids)

        triples = [triple.encode(self.tokenizer) for triple in kb_triples]

        encoded_corrupted_resp = self.tokenizer.encode(" " + corrupted_response, add_special_tokens=False)

        corrupt_indices = []
        if not self.is_inference:
            for old_phrase, new_phrase in example.swap_map.items():
                new_phrase_ids = self.tokenizer.encode(" " + new_phrase, add_special_tokens=False)
                idx = find_sublist(new_phrase_ids, encoded_corrupted_resp)
                assert idx is not None, (
                    f"at index {index}, phrase `{new_phrase}` {new_phrase_ids} cannot be found in the response to replace "
                    f"`{old_phrase}`: `{corrupted_response}` {encoded_corrupted_resp} - {example.swap_map}"
                )
                corrupt_indices.append((idx, new_phrase_ids))

        item_dict = self._build_from_segments(
            encoded_history,
            triples,
            encoded_corrupted_resp,
            corrupt_indices,
        )

        return item_dict

    def _build_from_segments(
        self,
        history: List[EncodedText],
        kb_triples: List[EncodedText],
        corrupted_response: EncodedText,
        corrupted_indices: List[Tuple[int, EncodedText]],
    ) -> Dict[str, List[int]]:
        """Builds a sequence of sentences pairs (corrupted history + mask).
        Example (sentence pair): `<s> d e f </s> </s> 1 2 3 </s>
        My sequence will be: '<s> history1 </s> subject </s> relation </s> obj </s> gold response </s>
        corrupted_response </s>
        I need later to add token level labels
        """
        tokens_history = []
        history_labels = []
        for i, utterance in enumerate(history):
            tokens_history.extend(utterance)
            history_labels.extend([-100] * len(utterance))

        tokens_triple = []
        triple_labels = []
        for triple in kb_triples:
            tokens_triple.extend(triple)
            tokens_triple.append(self.tokenizer.sep_token_id)
            triple_labels.extend([-100] * (len(triple) + 3))

        tokens = (
            [self.tokenizer.cls_token_id]
            + tokens_history
            + [self.tokenizer.sep_token_id]
            + tokens_triple
            + [self.tokenizer.sep_token_id]
            + corrupted_response
            + [self.tokenizer.eos_token_id]
        )

        output = dict(input_ids=tokens, attention_mask=[1] * len(tokens))

        if not self.is_inference:
            labels_corrupted_resp = [NERLabel.O.value[1]] * len(corrupted_response)
            for index, new_phrase_ids in corrupted_indices:
                labels_corrupted_resp[index] = NERLabel.B.value[1]
                for i in range(index + 1, index + len(new_phrase_ids)):
                    labels_corrupted_resp[i] = NERLabel.I.value[1]

            output["labels"] = (
                [-100] + history_labels + [-100] + triple_labels + [-100] + labels_corrupted_resp + [-100]
            )

        return output


def find_sublist(sublist, full_list) -> Optional[int]:
    """
    Finds a sublist inside a list and returns the index for the first item in the sublit from
    the big list
    """
    for i in range(len(full_list) - len(sublist) + 1):
        if full_list[i : i + len(sublist)] == sublist:
            return i
    return None


def text_swapping(text: str, swap_dict: Dict[str, str]) -> str:
    """
    Corrupts the original text with the randomly chosen entities.
    :param text: the original history.
    :param swap_dict: keys: original entities in the history; values: new entities to replace
    the old ones
    :return: str. A corrupted text with the new entities.
    """

    perturbed_text = text

    swap_dict = sorted(swap_dict.items(), key=lambda x: (len(x[0].split()), len(x[0])), reverse=True)
    for old_phrase, new_phrase in swap_dict:
        perturbed_text = re.sub(fr"\b{re.escape(old_phrase)}\b", f"{new_phrase}", perturbed_text)
        if old_phrase in perturbed_text:
            perturbed_text = (
                perturbed_text.replace(f" {old_phrase} ", f" {new_phrase} ")
                .replace(f"{old_phrase} ", f"{new_phrase} ")
                .replace(f" {old_phrase}", f" {new_phrase}")
            )

        assert new_phrase in perturbed_text, (
            f"`{old_phrase}` -> `{new_phrase}` replacement not successfully done in `{perturbed_text}` - "
            f"original text: `{text}` - swap dict: {swap_dict}"
        )

    return perturbed_text


def is_valid_swap(adversarial_candidate: str, original_texts: List[str]) -> bool:
    """
     Avoids replacing an entity with the same entity or its substring.
    :param adversarial_candidate: the selected adversarial entity
    :param original_text: the original entity in the text
    :return:
          Bool: True or False
    """
    normed_adversarial = adversarial_candidate.lower()
    return all(
        normed_adversarial != normed_text
        and normed_adversarial not in normed_text
        and normed_text not in normed_adversarial
        for normed_text in original_texts
    )


class CollatorCorruptResponse:
    def __init__(self, pad: int, label_pad: int = -100, as_tuple: bool = False):
        super().__init__()
        self.pad = pad
        self.label_pad = label_pad
        self.as_tuple = as_tuple
        self.fields = ("input_ids", "attention_mask", "token_type_ids", "labels")

    def __call__(self, batch):
        """Padding the instances within each batch. Batch is whatever you get from __get_item__().
        Batch is a list of dictionaries.
        """
        bsz = len(batch)
        padded_batch = {}
        # Go over each data point in the batch and pad each example.
        for name in self.fields:
            if not all(name in x for x in batch):
                continue

            if name == "labels":
                pad_token = self.label_pad
            else:
                pad_token = self.pad

            max_l = max(len(x[name]) for x in batch)
            # Fill the batches with padding tokens
            padded_field = np.full((bsz, max_l), pad_token, dtype=np.int64)
            # batch is a list of dictionaries
            for bidx, x in enumerate(batch):
                padded_field[bidx, : len(x[name])] = x[name]
            padded_batch[name] = torch.from_numpy(padded_field)

        return padded_batch
