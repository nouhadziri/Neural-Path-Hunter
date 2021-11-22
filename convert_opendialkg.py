import csv
import json
from argparse import ArgumentParser
from typing import Any, Dict, Iterable, Tuple

from tqdm import tqdm
import spacy

nlp = spacy.load("en")


def _tokenize(sent: str) -> str:
    return " ".join([tok.text for tok in nlp(sent)])


def read_csv(data_file: str) -> Iterable[Tuple[str, int]]:
    with open(data_file, "r") as f:
        reader = csv.reader(f, delimiter=",")
        next(reader)  # skip header row
        dialog_id = 0
        for i, row in enumerate(reader):
            dialog_id += 1
            dialogue, _, _ = row[0].strip(), row[1].strip(), row[2].strip()

            yield dialogue, dialog_id


def parse_message(dialogue: str, dialog_id: int) -> Iterable[Dict[str, Any]]:
    json_dialog = json.loads(dialogue)
    history = []
    metadata = {}
    for i, turn in enumerate(json_dialog):
        if i == 0:
            if "message" in turn:
                history.append(_tokenize(turn["message"]))
        else:
            if "metadata" in turn:
                if "path" in turn["metadata"]:
                    metadata = {
                        "paths": turn["metadata"]["path"][1],
                        "render": turn["metadata"]["path"][2],
                    }
            else:
                response = _tokenize(turn["message"])
                yield {
                    "history": history,
                    "response": response,
                    "speaker": turn["sender"],
                    "knowledge_base": metadata,
                    "dialogue_id": dialog_id,
                }

                metadata = {}
                history.append(response)


def convert(data_file: str, out_file: str):
    with open(out_file, "w") as out:
        for dialogue, dialog_id in tqdm(read_csv(data_file)):
            for utterance in parse_message(dialogue, dialog_id):
                out.write(json.dumps(utterance) + "\n")


def main():
    parser = ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input file")
    parser.add_argument("--out_file", type=str, required=True, help="Path to the output file")
    args = parser.parse_args()

    convert(args.input_file, args.out_file)


if __name__ == "__main__":
    main()
