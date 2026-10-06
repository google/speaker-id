# Finetuning Open-Source Models with Unsloth

This directory contains scripts and configurations used for finetuning open-source DiarizationLM models:

* **Gemma 4 E4B (4B, multi-domain, recommended):** https://huggingface.co/wq2012/DiarizationLM-Gemma-4-E4B-v1
* **Llama 3 8B (Fisher):** https://huggingface.co/google/DiarizationLM-8b-Fisher-v2
* **Llama 2 13B (Fisher):** https://huggingface.co/google/DiarizationLM-13b-Fisher-v1

## Files

* `config.py`: Default configuration (Llama 2 13B on Fisher). Modify this file to use your own data path.
* `config_llama3.py`: Configuration for Llama 3 8B (`google/DiarizationLM-8b-Fisher-v2`).
* `config_gemma4_e4b.py`: Configuration for Gemma 4 E4B (`wq2012/DiarizationLM-Gemma-4-E4B-v1`) trained across Fisher, Callhome, ICSI, and AMI.
* `dataprep.py`: Standard prompt-completion dataset builder (`hyp2ora` + `deg2ref`).
* `dataprep_locality.py`: Multi-domain dataset builder with Locality-Preserving Oracle speaker curation for multi-speaker meetings.
* `1_finetune.py`: Run this script on a machine with GPU to finetune the model.
* `2_export.py`: Export the model once finetuning is completed.
* `3_batch_inference.py`: Run batch inference of the finetuned model on evaluation data to evaluate it later.
* `4_eval.py`: Compute evaluation metrics based on inference outputs.

We also provide example usage of the finetuned models in `example_usage.py` (Llama) and `example_usage_gemma4.py` (`wq2012/DiarizationLM-Gemma-4-E4B-v1`).