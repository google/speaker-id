"""To use this config, in all other python scripts, change:

import config

to

import config_llama3 as config
"""
# DataPrep
TRAINING_INPUT = {
    "FISHER": ("/YOUT_DATA_PATH/FISHER_ENGLISH_TRAIN_FULL.json", 1),
}
EMIT_INPUT_LENGTH = 6000
EMIT_TARGET_LENGTH = 6000
PROMPT_PREFIX = ""
PROMPT_SUFFIX = " --> "
COMPLETION_SUFFIX = " [eod]"

# Train
RESUME_FROM_CHECKPOINT = False
MODEL_NAME = "unsloth/llama-3-8b-bnb-4bit"
LORA_RANK = 256
MAX_SEQ_LENGTH = 4096
MAX_STEPS = 25400
DATA_NAME = "_".join(TRAINING_INPUT.keys())
MODEL_ID = f"{MODEL_NAME}_{DATA_NAME}_LORA{LORA_RANK}_LEN{MAX_SEQ_LENGTH}"

# Export
CHECKPOINT = 25400

# Inference for evaluation
EVAL_INPUTS = {
    "FISHER": "/YOUT_DATA_PATH/FISHER_ENGLISH_TEST_FULL.json",
    "CALLHOME": "/YOUT_DATA_PATH/CALLHOME_ENGLISH_TEST_FULL.json",
}
