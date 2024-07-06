# Finetuning llama-2-13b with unsloth

This directory contains scripts used for finetuning https://huggingface.co/google/DiarizationLM-13b-Fisher-v1

* config.py: Modify this file to use your own data path.
* 1_finetune.py: Run this script on a machine with GPU to finetune the model.
* 2_export.py: Export the model once finetuning is completed.
* 3_batch_inference.py: Run batch inference of the finetuned model on evaluation data to evaluate it later.
* 4_eval.py: Compute evaluation metrics based on inference outputs.

We also provide example usage of the finetune model in example_usage.py.