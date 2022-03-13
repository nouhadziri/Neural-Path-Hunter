import json
import logging
import os
import shutil
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List

import numpy as np
import pytorch_lightning as pl
import spacy
import torch
from pytorch_lightning import loggers as pl_loggers
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.nn import CrossEntropyLoss

from .data import CollatorCorruptResponse, NamedEntity, NERLabel, ResponseCorruptedDataset
from ..lightning_base import BaseTransformer, add_generic_args, generic_train
from ..log_utils import is_wandb_available, authorize_wandb

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("token_hallucination")


class NERTransformer(BaseTransformer):
    """
    A training module for NER. See BaseTransformer for the core options.
    """

    mode = "token-classification"

    def __init__(self, hparams):
        if type(hparams) == dict:
            hparams = Namespace(**hparams)

        self.pad_token_label_id = CrossEntropyLoss().ignore_index
        super().__init__(hparams, self.mode, num_labels=len(NERLabel), return_dict=True)
        self.nlp = spacy.load("en_core_web_sm")
        self.tagger = None

    def forward(self, **inputs):
        return self.model(**inputs)

    def _get_entities(self, text: str) -> List[NamedEntity]:
        annotated_text = self.nlp(text)
        return [NamedEntity(e.text.strip(), e.label_) for e in annotated_text.ents]

    def training_step(self, batch, batch_num):
        " Compute loss and log."
        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch.get("attention_mask", None),
            "labels": batch["labels"],
        }

        if self.config.model_type != "distilbert":
            inputs["token_type_ids"] = (
                batch.get("token_type_ids", None) if self.config.model_type in ["bert", "xlnet"] else None
            )  # XLM and RoBERTa don"t use token_type_ids

        outputs = self(**inputs)
        loss = outputs[0]
        return {"loss": loss}

    def get_dataset(self, mode: str, data_path: str, **kwargs):
        dataset = ResponseCorruptedDataset(
            self.tokenizer,
            data_path,
            self.hparams.output_dir,
            self._get_entities,
            self.hparams.graph_file,
            self.hparams.output_path,
            self.hparams.num_swaps,
            self.hparams.num_corrupted_samples,
            self.hparams.max_attempts,
            self.hparams.max_history,
            self.hparams.max_seq_length,
            self.hparams.entity_pickle_file,
            self.hparams.overwrite_corrupted_data,
            is_inference=mode.lower() == "infer",
            use_groundtruth_response=self.hparams.use_groundtruth_response,
        )

        logger.info(f"{mode} dataset size: {len(dataset)}")

        return dataset

    def get_collator(self, mode: str):
        return CollatorCorruptResponse(self.tokenizer.pad_token_id)

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def validation_step(self, batch, batch_nb):
        """Compute validation""" ""
        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch.get("attention_mask", None),
            "labels": batch.get("labels", None),
        }
        if self.config.model_type != "distilbert":
            inputs["token_type_ids"] = (
                batch.get("token_type_ids", None) if self.config.model_type in ["bert", "xlnet"] else None
            )  # XLM and RoBERTa don"t use token_type_ids
        output = self(**inputs)
        tmp_eval_loss = output.loss
        preds = output.logits.detach().cpu().numpy()

        if "labels" in batch:
            out_label_ids = inputs["labels"].detach().cpu().numpy()
            step_result = dict(val_loss=tmp_eval_loss.detach().cpu(), pred=preds, target=out_label_ids)
        else:
            step_result = dict(pred=preds)

        return step_result

    def _eval_end(self, outputs):
        "Evaluation called for both Val and Test"
        label_map = NERLabel.map()
        out_label_list = []
        preds_list = []

        if any(x.get("target", None) is None for x in outputs):
            for x in outputs:
                batch_preds = np.argmax(x["pred"], axis=2)
                preds = [[] for _ in range(batch_preds.shape[0])]

                for i in range(batch_preds.shape[0]):
                    for j in range(batch_preds.shape[1]):
                        preds[i].append(label_map[batch_preds[i][j]])
                preds_list.extend(preds)

            results = {}
        else:
            val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean()

            for x in outputs:
                batch_preds = np.argmax(x["pred"], axis=2)
                batch_label_ids = x["target"]

                preds = [[] for _ in range(batch_label_ids.shape[0])]

                out_labels = [[] for _ in range(batch_label_ids.shape[0])]

                for i in range(batch_label_ids.shape[0]):
                    for j in range(batch_label_ids.shape[1]):
                        if batch_label_ids[i, j] != self.pad_token_label_id:
                            out_labels[i].append(label_map[batch_label_ids[i][j]])
                            preds[i].append(label_map[batch_preds[i][j]])

                out_label_list.extend(out_labels)
                preds_list.extend(preds)

            results = dict(
                val_loss=val_loss_mean,
                accuracy_score=accuracy_score(out_label_list, preds_list),
                precision=precision_score(out_label_list, preds_list),
                recall=recall_score(out_label_list, preds_list),
                f1=f1_score(out_label_list, preds_list),
            )

        return results, preds_list, out_label_list

    def validation_epoch_end(self, outputs):
        # when stable
        results, *_ = self._eval_end(outputs)
        for k, v in results.items():
            self.log(f"valid/{k}", v, prog_bar=False)
            self.log(k, v, logger=False, prog_bar=True)

    def test_epoch_end(self, outputs):
        # updating to test_epoch_end instead of deprecated test_end
        results, predictions, _ = self._eval_end(outputs)

        # Converting to the dict required by pl
        # https://github.com/PyTorchLightning/pytorch-lightning/blob/master/\
        # pytorch_lightning/trainer/logging.py#L139
        if "val_loss" in results:
            results["loss"] = results.pop("val_loss")

        for k, v in results.items():
            self.log(f"test/{k}", v, prog_bar=False)
            self.log(k, v, logger=False, prog_bar=True)

        self._save_predictions(predictions, unlabelled="loss" not in results)

    def _save_predictions(self, predictions, unlabelled: bool = False):
        dataset = ResponseCorruptedDataset(
            self.tokenizer,
            self.hparams.test_dataset_path,
            self.hparams.output_dir,
            self._get_entities,
            # self._get_entities_not_spacy,
            self.hparams.graph_file,
            self.hparams.output_path,
            self.hparams.num_swaps,
            self.hparams.num_corrupted_samples,
            self.hparams.max_attempts,
            max_history=self.hparams.max_history,
            max_seq_length=self.hparams.max_seq_length,
            is_inference=unlabelled,
            use_groundtruth_response=self.hparams.use_groundtruth_response,
        )

        preds_file = os.path.join(
            self.hparams.output_dir, f"preds_{Path(dataset._ner_data_path).with_suffix('').name}.jsonl"
        )

        assert len(dataset.data) == len(predictions), "number of predictions must match test data size"
        num_predictions = 0
        num_samples_with_halluc = 0

        with open(preds_file, "w") as writer:
            for i, pred in enumerate(predictions):
                ex = dataset.data[i].asdict()
                corrupted_response = self.tokenizer.tokenize(" " + ex["corrupted_response"])
                if not ex["swaps"]:
                    ex["generated_response"] = ex.pop("corrupted_response")
                    pred = pred[: len(dataset[i]["input_ids"])][-len(corrupted_response) - 1 : -1]

                assert len(corrupted_response) == len(pred), (
                    f"discrepancy at index {i}, corrupted response length {len(corrupted_response)} != "
                    f"prediction length {len(pred)}: {corrupted_response} vs. {pred}"
                )

                contiguous_seqs = []
                pred_label = []
                for j, p in enumerate(pred):
                    if p.startswith("B"):
                        if pred_label:
                            contiguous_seqs.append([corrupted_response[k] for k in pred_label])
                        pred_label = [j]
                    elif p.startswith("I"):
                        pred_label.append(j)

                if pred_label:
                    contiguous_seqs.append([corrupted_response[k] for k in pred_label])

                ex["raw_prediction"] = pred
                ex["prediction"] = [self.tokenizer.convert_tokens_to_string(seq).strip() for seq in contiguous_seqs]
                num_predictions += len(ex["prediction"])
                if ex["prediction"]:
                    num_samples_with_halluc += 1

                if ex["swaps"]:
                    ex["results"] = {
                        "false_positives": [],
                        "false_negatives": [],
                    }

                    normed_gold_labels = set(sw["new"].lower() for sw in ex["swaps"])
                    normed_preds = set(p.lower() for p in ex["prediction"])

                    for p in ex["prediction"]:
                        if p.lower() not in normed_gold_labels:
                            ex["results"]["false_positives"].append(p)

                    for g in set(sw["new"] for sw in ex["swaps"]):
                        if g.lower() not in normed_preds:
                            ex["results"]["false_negatives"].append(g)
                else:
                    ex.pop("swaps")

                writer.write(json.dumps(ex) + "\n")

        print(
            f"Avg #predicted hallucination: {num_predictions / len(predictions):.1f} = "
            f"{num_predictions} / {len(predictions)}"
        )
        print(
            f"#dialogues with hallucination: {100. * num_samples_with_halluc / len(predictions):.1f}% = "
            f"{num_samples_with_halluc} / {len(predictions)}"
        )

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        # Add NER specific options
        BaseTransformer.add_model_specific_args(parser, root_dir)
        parser.add_argument(
            "--num_masks",
            type=int,
            default=2,
            help="Number of masks to apply on a sentence.",
        )
        parser.add_argument(
            "--num_swaps",
            type=int,
            default=2,
            help="Number of swaps.",
        )
        parser.add_argument(
            "--num_corrupted_samples",
            type=int,
            default=1,
            help="Number of swaps.",
        )
        parser.add_argument(
            "--max_attempts",
            type=int,
            default=3,
            help="Number of swaps.",
        )
        parser.add_argument(
            "--topk_pred",
            type=int,
            default=1,
            help="Number of topk predicted tokens for one mask.",
        )
        parser.add_argument(
            "--entity_pickle_file",
            type=str,
            default=None,
            help="Pickle file containing a dictionary that categorizes entities by their types",
        )
        parser.add_argument(
            "--overwrite_corrupted_data",
            action="store_true",
            default=False,
            help="Whether to overwrite the corrupted data or use the already-created one",
        )
        parser.add_argument(
            "--use_groundtruth_response",
            action="store_true",
            default=False,
            help="During inference, whether to use ground-truth response as input, instead of generated response",
        )

        parser.add_argument(
            "--graph_file",
            type=str,
            default=None,
            help="Path to the graph file.",
        )

        parser.add_argument(
            "--output_path",
            type=str,
            default=None,
            help="Path to the output file.",
        )

        return parser


def main():
    parser = ArgumentParser()
    add_generic_args(parser)
    NERTransformer.add_model_specific_args(parser, os.getcwd())

    args = parser.parse_args()

    if args.output_dir is None:
        if os.path.isdir(args.model_name_or_path):
            args.output_dir = args.model_name_or_path
        else:
            args.output_dir = "./checkpoints"

    odir = Path(args.output_dir)
    if args.do_train and args.overwrite_output_dir and odir.exists():
        shutil.rmtree(odir)
    odir.mkdir(parents=True, exist_ok=True)

    model = NERTransformer(args)

    extra_callbacks = []
    if args.do_train and args.patience > 0:
        extra_callbacks.append(
            pl.callbacks.EarlyStopping(
                monitor="valid/f1",
                min_delta=args.min_delta,
                patience=args.patience,
                verbose=False,
                mode="max",
            )
        )

    if args.wandb and is_wandb_available():
        logger_callback = pl_loggers.WandbLogger(
            project="Token-Hallucination",
            name="Finetune-{}".format(args.namestr),
            anonymous=not authorize_wandb(),
        )
    else:
        logger_callback = pl_loggers.TensorBoardLogger(
            save_dir=args.output_dir,
            name="train_logs" if args.do_train else "valid_logs",
            default_hp_metric=False,
        )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.output_dir,
        prefix="checkpoint",
        monitor="valid/f1",
        mode="max",
        save_top_k=1,
    )

    trainer_kwargs = {}

    trainer = generic_train(
        model,
        args,
        logger_callback,
        extra_callbacks,
        checkpoint_callback,
        **trainer_kwargs,
    )

    if args.do_test or args.do_infer:
        if args.do_train:
            args.output_dir = os.path.join(args.output_dir, "best_model")
            logger.info(f"Reload model from `{args.output_dir}` ...")
            model = NERTransformer(args)

        if args.do_test:
            trainer.test(model)

        if args.do_infer:
            trainer.test(
                model,
                model.get_dataloader("infer", args.eval_batch_size, shuffle=False, data_path=args.test_dataset_path),
            )


if __name__ == "__main__":
    main()
