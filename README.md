# Neural Path Hunter (NPH)

This repository hosts the implementation of the paper "[Neural Path Hunter: Reducing Hallucination in Dialogue Systems via
Path Grounding](https://aclanthology.org/2021.emnlp-main.168.pdf)", published in EMNLP'21.

Dialogue systems powered by large pre-trained language models (LM) exhibit an innate ability to deliver fluent and natural-looking responses.
Despite their impressive generation performance, these models can often generate factually incorrect statements impeding their widespread adoption.
In this paper, we focus on the task of improving the faithfulness -- and thus reduce hallucination -- of Neural Dialogue Systems to known facts supplied by a Knowledge Graph (KG). 
We propose Neural Path Hunter which follows a generate-then-refine strategy whereby a generated response is amended using the k-hop subgraph of a KG. 
Our proposed model can easily be applied to any dialogue generated responses without retraining the model. We empirically validate our proposed approach on the OpenDialKG dataset against a suite of metrics and report a relative improvement of faithfulness over dialogue responses by 20.35% based on FeQA (Durmus et al., 2020).

## Prerequisites

Install the dependencies:
```
pip install -r requirements.txt
python -m spacy link en_core_web_sm en
```

## Dataset
Download the raw data [OpenDialKG](https://github.com/facebookresearch/opendialkg) and then run the following command to process the data:

```
python convert_opendialkg.py --input_file <PATH_OpenDialKG_CSV> --output_file <PATH_OUTPUT>
 ```
 
## Train NPH
First, download the [graph embeddings](https://drive.google.com/drive/folders/1KzjCq0-8K1pqi1TFfsEC3iKiaK-2oL1I?usp=sharing)
and then run the following command to train NPH:
```
python -m dialkg.mask_refine --model_name_or_path gpt2 --train_dataset_path <PATH_TRAIN_DATA> --eval_dataset_path <PATH_DEV_DATA>  --do_train --output_dir <PATH_OUTPUT_DIR> --num_train_epochs <NUM_EPOCH> --train_batch_size 16 --eval_batch_size 16  --kge <PATH_GRAPH_EMBED> --graph_file <PATH_TO_GRAPH_FILE> --pad_to_multiple_of 8 --max_adjacents 100 --patience 10 --max_history 3
```

<pre>
    --model_name_or_path     default: gpt2
    --train_dataset_path     OpenDialKG train data
    --eval_dataset_path      OpenDialKG dev data
    --output_dir             Path to the output directory
    --kge                    Path to the graph embeddings
    --graph_file             Path to the graph file  (opendialkg_triples.txt)
    --num_train_epochs       default: 3
    --train_batch_size       default: 16
    --eval_batch_size        default: 16
    --pad_to_multiple_of     default: 8
    --max_adjacents          default: 100
    --patience               default: 10
    --max_history            default: 3      
</pre>

## NPH Refinement
To fix the hallucinated responses,  download the [token-level hallucination critic]() and run the following command:
```
python -m dialkg.mask_refine --refine_only --model_name_or_path <PATH_TRAINED_NPH>  --kge <PATH_GRAPH_EMBED> --graph_file <PATH_TO_GRAPH_FILE> --pad_to_multiple_of 8 --halluc_classifier <PATH_HALLUCINATION_CLASSIFIER> --generated_path <PATH_TO_RESPONSES_TO_FIX> --do_generate  --eval_batch_size 16 
```

<pre>
    --model_name_or_path     Path to the trained NPH model
    --halluc_classifier      Path to the entity-level hallucination classifier
    --generated_path         Path to the JSONL-formatted generated responses which we aim to refine
</pre>

Here is a sample of the file containing machine-generated responses:
```
{
   "history":[
      "Do you know any information on the book Crescendo. It has been recommended to me."
   ],
   "response":"Yes, it's written by Becca Fitzpatrick and it is a young-adult fiction. ",
   "speaker":"assistant",
   "dialogue_id":12421,
   "knowledge_base":[
      [
         "Crescendo",
         "written_by",
         "Becca Fitzpatrick"
      ]
   ],
   "generated_response":[
      "Yes! Crescendo is a Romance novel by Becca Fitzpatrick"
   ]
}
```

## Citation
Please cite our work if you use it in your research:
```angular2html
@inproceedings{dziri-etal-2021-neural,
    title = "Neural Path Hunter: Reducing Hallucination in Dialogue Systems via Path Grounding",
    author = {Dziri, Nouha  and
      Madotto, Andrea  and
      Za{\"\i}ane, Osmar  and
      Bose, Avishek Joey},
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.168",
    pages = "2197--2214",
    abstract = "Dialogue systems powered by large pre-trained language models exhibit an innate ability to deliver fluent and natural-sounding responses. Despite their impressive performance, these models are fitful and can often generate factually incorrect statements impeding their widespread adoption. In this paper, we focus on the task of improving faithfulness and reducing hallucination of neural dialogue systems to known facts supplied by a Knowledge Graph (KG). We propose Neural Path Hunter which follows a generate-then-refine strategy whereby a generated response is amended using the KG. Neural Path Hunter leverages a separate token-level fact critic to identify plausible sources of hallucination followed by a refinement stage that retrieves correct entities by crafting a query signal that is propagated over a k-hop subgraph. We empirically validate our proposed approach on the OpenDialKG dataset (Moon et al., 2019) against a suite of metrics and report a relative improvement of faithfulness over dialogue responses by 20.35{\%} based on FeQA (Durmus et al., 2020). The code is available at https://github.com/nouhadziri/Neural-Path-Hunter.",
}

```


