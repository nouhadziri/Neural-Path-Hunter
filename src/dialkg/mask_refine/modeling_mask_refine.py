import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from torch_scatter import scatter_mean
from typing import Dict, List, Optional, Union

from ..utils import masked_log_softmax

WEIGHTS_NAME = "model.bin"


@dataclass
class MaskRefineOutput:
    logits: List[torch.Tensor]
    selected_cands: torch.Tensor
    topK_cands: torch.Tensor
    topK_rels: torch.Tensor
    topK_pivots: Optional[torch.Tensor] = None
    topK_pivot_fields: Optional[torch.Tensor] = None
    losses: Optional[torch.Tensor] = None
    accuracies: Optional[torch.Tensor] = None
    l_mr: Optional[torch.Tensor] = None
    l_mrr: Optional[torch.Tensor] = None
    l_hits1: Optional[torch.Tensor] = None
    l_hits3: Optional[torch.Tensor] = None
    l_hits10: Optional[torch.Tensor] = None

    @property
    def loss(self):
        return torch.mean(self.losses) if self.losses is not None else None

    @property
    def accuracy(self):
        return torch.mean(self.accuracies) if self.accuracies is not None else None

    @property
    def mean_reciprocal_rank(self):
        return torch.mean(self.l_mrr) if self.l_mrr is not None else None

    @property
    def mean_rank(self):
        return torch.mean(self.l_mr) if self.l_mr is not None else None

    @property
    def hits1(self):
        return torch.mean(self.l_hits1) if self.l_hits1 is not None else None

    @property
    def hits3(self):
        return torch.mean(self.l_hits3) if self.l_hits3 is not None else None

    @property
    def hits10(self):
        return torch.mean(self.l_hits10) if self.l_hits10 is not None else None


def batch_cos(a, b, eps=1e-8):
    """
    https://stackoverflow.com/a/58144658
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=-1)[:, :, None], b.norm(dim=-1)[:, :, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.bmm(a_norm, b_norm.transpose(1, 2))
    return sim_mt


def tanh_clip(x, clip_val=None):
    if clip_val is not None:
        return clip_val * torch.tanh((1.0 / clip_val) * x)
    else:
        return x


def calc_regulaizer_coef(scores, regularizer_coef: float = 4e-2):
    return regularizer_coef * (scores ** 2.0).mean()


class MaskRefineModel(nn.Module):
    def __init__(
        self,
        lm_model,
        mlm_model,
        ete: Optional[np.array] = None,
        rte: Optional[np.array] = None,
        num_nodes: int = 0,
        num_rels: int = 0,
        inbatch_negatives: bool = False,
        stabilize: bool = True,
        regularizer_coef: float = 4e-2,
        clip_val: float = 10.0,
        topK: int = 15,
    ):
        super().__init__()
        self.lm_model = lm_model
        self.mlm_model = mlm_model

        if ete is not None:
            self.ete = nn.Embedding.from_pretrained(torch.Tensor(ete), freeze=False)
        else:
            assert num_nodes > 0
            self.ete = nn.Embedding(num_nodes, self.lm_model.config.n_embd)
        self.kg_ent_head = nn.Linear(self.ete.embedding_dim, self.lm_model.config.n_embd, bias=False)

        if rte is not None:
            self.rte = nn.Embedding.from_pretrained(torch.Tensor(rte), freeze=False)
        else:
            assert num_rels > 0
            self.rte = nn.Embedding(num_rels, self.lm_model.config.n_embd)
        self.kg_rel_head = nn.Linear(self.rte.embedding_dim, self.lm_model.config.n_embd, bias=False)

        if self.mlm_model is not None:
            self.ln_transform = torch.nn.Linear(
                self.lm_model.config.n_embd + self.mlm_model.config.hidden_size, self.lm_model.config.n_embd, bias=False
            )
            self.init_ln_transform = torch.nn.Linear(
                self.mlm_model.config.hidden_size, self.lm_model.config.n_embd, bias=False
            )
        else:
            self.lm_ete = nn.Embedding(self.ete.num_embeddings, self.lm_model.config.n_embd)
            self.ln_transform = torch.nn.Linear(
                self.lm_model.config.n_embd + self.lm_ete.embedding_dim, self.lm_model.config.n_embd, bias=False
            )
            self.init_ln_transform = None

        self.inbatch_negatives = inbatch_negatives

        self.stabilize = stabilize
        self.regularizer_coef = regularizer_coef
        self.clip_val = clip_val

        self.topK = topK

    def save_pretrained(self, save_directory):
        if os.path.isfile(save_directory):
            return

        os.makedirs(save_directory, exist_ok=True)

        # Only save the model itself if we are using distributed training
        model_to_save = self.module if hasattr(self, "module") else self

        state_dict = model_to_save.state_dict()
        state_dict = {
            k: v for k, v in state_dict.items() if not k.startswith("mlm_model") and not k.startswith("lm_model")
        }

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, WEIGHTS_NAME)
        torch.save(state_dict, output_model_file)

    @classmethod
    def _from_pretrained(
        cls, pretrained_model_name_or_path, lm_model, mlm_model, num_nodes: int, num_rels: int, **kwargs
    ):
        if not os.path.isfile(pretrained_model_name_or_path):
            raise FileNotFoundError(f"{pretrained_model_name_or_path}")

        model_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
        model = MaskRefineModel(lm_model, mlm_model, num_nodes=num_nodes, num_rels=num_rels, **kwargs)
        model.load_state_dict(torch.load(model_file))
        return model

    @classmethod
    def new_or_from_pretrained(
        cls,
        pretrained_model_name_or_path,
        lm_model,
        mlm_model,
        ete: np.array,
        rte: np.array,
        inbatch_negatives: bool = False,
        stabilize: bool = True,
        regularizer_coef: float = 4e-2,
        clip_val: float = 10.0,
        topK: int = 15,
    ):
        weights_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
        if os.path.isfile(weights_file):
            return cls._from_pretrained(
                pretrained_model_name_or_path,
                lm_model,
                mlm_model,
                num_nodes=ete.shape[0],
                num_rels=rte.shape[0],
                inbatch_negatives=inbatch_negatives,
                stabilize=stabilize,
                regularizer_coef=regularizer_coef,
                clip_val=clip_val,
                topK=topK,
            )
        else:
            return MaskRefineModel(
                lm_model,
                mlm_model,
                ete,
                rte,
                inbatch_negatives=inbatch_negatives,
                stabilize=stabilize,
                regularizer_coef=regularizer_coef,
                clip_val=clip_val,
                topK=topK,
            )

    def forward(
        self,
        mlm_inputs: Union[Dict[str, torch.Tensor], torch.LongTensor],
        lm_input_ids: torch.LongTensor,
        lm_attention_masks: torch.LongTensor,
        candidate_ids: torch.LongTensor,
        candidate_rels: torch.LongTensor,
        pivot_ids: torch.LongTensor,
        mlm_entity_mask: Optional[torch.LongTensor] = None,
        pivot_fields: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        label_indices: Optional[torch.LongTensor] = None,
        inbatch_negatives: Optional[bool] = None,
    ):


        if self.mlm_model is not None:
            mlm_output = self.mlm_model(**mlm_inputs)
            mlm_last_layer_hidden_states = mlm_output.hidden_states[-1]
            mlm_entity_reps = scatter_mean(mlm_last_layer_hidden_states, mlm_entity_mask, dim=1)[:, 1:, :]
        else:
            mlm_entity_reps = None

        B = lm_input_ids.shape[0]
        lm_range_tensor = torch.arange(B)

        prev_lm_entity_reps = None
        nb_masked_entities = lm_attention_masks.shape[1]

        nce_losses = []
        nce_accuracies = []
        all_logits = []
        selected_cands, topK_cands, topK_rels = [], [], []
        topK_pivots, topK_pivot_fields = [], []
        l_mrr, l_mr, l_hits1, l_hits3, l_hits10 = [], [], [], [], []

        for j in range(nb_masked_entities):
            lm_inputs_embeds = self.lm_model.transformer.wte(lm_input_ids)

            for k in range(j + 1):
                entity_index = (lm_attention_masks[:, k, :] == 1).int().sum(-1) - 1
                if self.mlm_model is None:
                    entity_reps = self.lm_ete(mlm_inputs[:, k])
                    if j == k and prev_lm_entity_reps is not None:
                        entity_reps = self.ln_transform(torch.cat([prev_lm_entity_reps, entity_reps], dim=-1))
                    else:
                        entity_reps = self.kg_ent_head(entity_reps)
                else:
                    entity_reps = mlm_entity_reps[:, k, :]
                    if j == k and prev_lm_entity_reps is not None:
                        entity_reps = self.ln_transform(torch.cat([prev_lm_entity_reps, entity_reps], dim=-1))
                    else:
                        entity_reps = self.init_ln_transform(entity_reps)
                lm_inputs_embeds[lm_range_tensor, entity_index, :] = entity_reps

            # lm_output = B * hidden_state
            lm_output = self.lm_model(inputs_embeds=lm_inputs_embeds)

            # lm_hidden_states = B * hidden_states
            lm_hidden_states = lm_output.hidden_states[-1][lm_range_tensor, entity_index]
            prev_lm_entity_reps = lm_hidden_states

            neighbor_embeds = self.kg_ent_head(self.ete(candidate_ids[:, j, ...]))
            pivot_embeds = self.kg_ent_head(self.ete(pivot_ids[:, j]))

            if len(pivot_embeds.shape) > 2:
                lm_hidden_states = lm_hidden_states.unsqueeze(1)

            object_embeds = pivot_embeds * lm_hidden_states
            H = object_embeds.shape[-1]
            N = neighbor_embeds.shape[2 if len(neighbor_embeds.shape) > 3 else 1]

            scores = torch.bmm(
                neighbor_embeds.view(-1, N, H), object_embeds.view(-1, H).unsqueeze(1).transpose(1, 2)
            ).squeeze(-1) / np.sqrt(H)

            scores = scores.reshape(B, -1)

            if inbatch_negatives or self.inbatch_negatives:
                inbatch_mask = (
                    (1 - torch.eye(B, device=scores.device))
                    .unsqueeze(-1)
                    .expand(B, B, neighbor_embeds.shape[1])
                    .reshape(B, -1)
                )
                mask = torch.cat([torch.ones_like(scores), inbatch_mask], dim=-1)

                inbatch_scores = torch.mm(
                    object_embeds, neighbor_embeds.view(-1, neighbor_embeds.shape[-1]).T
                ) / np.sqrt(H)
                scores = masked_log_softmax(torch.cat([scores, inbatch_scores], dim=-1), mask)
                current_candidate_ids = torch.cat(
                    [candidate_ids[:, j], candidate_ids[:, j].reshape(1, -1).repeat(B, 1)], dim=-1
                )
                current_candidate_rels = torch.cat(
                    [candidate_rels[:, j], candidate_rels[:, j].reshape(1, -1).repeat(B, 1)], dim=-1
                )
            else:
                scores = F.log_softmax(scores, dim=-1)
                current_candidate_ids = candidate_ids[:, j].reshape(B, -1)
                current_candidate_rels = candidate_rels[:, j].reshape(B, -1)

            if self.stabilize:
                scores = tanh_clip(scores, self.clip_val)
                reg = calc_regulaizer_coef(scores, self.regularizer_coef)
            else:
                reg = 0.0

            _, max_score_indices = torch.max(scores, dim=1)
            selected_cands.append(current_candidate_ids[torch.arange(B), max_score_indices].unsqueeze(0))

            topK_scores, topK_score_indices = torch.topk(scores, k=min(self.topK, scores.shape[1]), dim=1, largest=True)
            topK_cands.append(torch.gather(current_candidate_ids, dim=-1, index=topK_score_indices).unsqueeze(0))
            topK_rels.append(torch.gather(current_candidate_rels, dim=-1, index=topK_score_indices).unsqueeze(0))

            if pivot_fields is not None:
                topK_pivot_fields.append(
                    torch.gather(pivot_fields[:, j], dim=-1, index=topK_score_indices // N).unsqueeze(0)
                )

                topK_pivots.append(
                    torch.gather(pivot_ids[:, j], dim=-1, index=topK_score_indices // N).unsqueeze(0)
                )

            if label_indices is not None:
                current_labels = label_indices[:, j]
                accuracy = (max_score_indices == current_labels).float().sum() / (current_labels >= 0).int().sum()

                nonpad_labels = torch.masked_select(current_labels, current_labels.ge(0))
                ranking_metrics = self.compute_ranks(scores[current_labels >= 0, :], nonpad_labels)

                l_mr.append(torch.FloatTensor([ranking_metrics["MR"]]))
                l_mrr.append(torch.FloatTensor([ranking_metrics["MRR"]]))
                l_hits1.append(torch.FloatTensor([ranking_metrics["HITS@1"]]))
                l_hits3.append(torch.FloatTensor([ranking_metrics["HITS@3"]]))
                l_hits10.append(torch.FloatTensor([ranking_metrics["HITS@10"]]))
                nce_accuracies.append(accuracy)

                pos_scores = scores[current_labels >= 0, nonpad_labels]
                nce_loss = -pos_scores.mean() + reg
                nce_losses.append(nce_loss)

            all_logits.append(scores)

        return MaskRefineOutput(
            all_logits,
            torch.vstack(selected_cands).T,
            torch.vstack(topK_cands).transpose(1, 0),
            torch.vstack(topK_rels).transpose(1, 0),
            torch.vstack(topK_pivots).transpose(1, 0) if topK_pivots else None,
            torch.vstack(topK_pivot_fields).transpose(1, 0) if topK_pivot_fields else None,
            torch.vstack(nce_losses) if nce_losses else None,
            torch.vstack(nce_accuracies) if nce_accuracies else None,
            torch.vstack(l_mr) if l_mr else None,
            torch.vstack(l_mrr) if l_mrr else None,
            torch.vstack(l_hits1) if l_hits1 else None,
            torch.vstack(l_hits3) if l_hits3 else None,
            torch.vstack(l_hits10) if l_hits10 else None,
        )

    def compute_ranks(self, scores, current_labels):
        argsort = torch.argsort(scores, dim=1, descending=True)
        batch_size = scores.shape[0]
        logs = []
        for i in range(batch_size):
            # Notice that argsort is not rankingc
            # Returns the rank of the current_labels in argsort
            ranking = torch.nonzero(argsort[i, :] == current_labels[i], as_tuple=False)
            assert ranking.shape[0] == 1

            # ranking + 1 is the true ranking used in evaluation metrics
            ranking = 1 + ranking.item()
            logs.append(
                {
                    "MRR": 1.0 / ranking,
                    "MR": float(ranking),
                    "HITS@1": 1.0 if ranking <= 1 else 0.0,
                    "HITS@3": 1.0 if ranking <= 3 else 0.0,
                    "HITS@10": 1.0 if ranking <= 10 else 0.0,
                }
            )
        metrics = {}
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs]) / len(logs)

        return metrics
