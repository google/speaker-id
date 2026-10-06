"""Configuration for Gemma 4 E4B (wq2012/DiarizationLM-Gemma-4-E4B-v1).

To use this config, in other python scripts, change:

import config

to

import config_gemma4_e4b as config
"""

# DataPrep: Multi-domain mixture across Fisher, Callhome, ICSI, and AMI
TRAINING_INPUT = {
    "FISHER": ("/YOUR_DATA_PATH/FISHER_ENGLISH_TRAIN_FULL.json", 1),
    "CALLHOME": ("/YOUR_DATA_PATH/CALLHOME_ENGLISH_TRAIN_FULL.json", 5),
    "ICSI": ("/YOUR_DATA_PATH/ICSI_TRAIN_FULL.json", 10),
    "AMI": ("/YOUR_DATA_PATH/AMI_TRAIN_WORD_FULL.json", 10),
}
EMIT_INPUT_LENGTH = 4000
EMIT_TARGET_LENGTH = 4000
PROMPT_PREFIX = ""
PROMPT_SUFFIX = " --> "
COMPLETION_SUFFIX = " [eod]"
MAX_LOCAL_SPAN_WORDS = 5

# Train
RESUME_FROM_CHECKPOINT = False
MODEL_NAME = "google/gemma-4-E4B"
LORA_RANK = 256
MAX_SEQ_LENGTH = 2560
MAX_STEPS = 10000
DATA_NAME = "_".join(TRAINING_INPUT.keys())
MODEL_ID = (
    f"{MODEL_NAME.replace('/', '_')}_{DATA_NAME}_"
    f"LORA{LORA_RANK}_LEN{MAX_SEQ_LENGTH}"
)

# Export
CHECKPOINT = 10000

# Inference for evaluation across all 4 benchmarks
EVAL_INPUTS = {
    "FISHER": "/YOUR_DATA_PATH/FISHER_ENGLISH_TEST_FULL.json",
    "CALLHOME": "/YOUR_DATA_PATH/CALLHOME_ENGLISH_TEST_FULL.json",
    "ICSI": "/YOUR_DATA_PATH/ICSI_TEST_FULL.json",
    "AMI": "/YOUR_DATA_PATH/AMI_TEST_WORD_FULL.json",
}
