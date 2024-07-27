"""Finetune the model.

For GCP, we sometimes need to run `export PATH=$PATH:/sbin`
before running this script.
"""

import config
import dataprep
import torch
from transformers import TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported


def run_training() -> None:
  ############################################################################
  # Get dataset
  ############################################################################
  dataset = dataprep.build_dataset()

  ############################################################################
  # Get model
  ############################################################################
  model, tokenizer = FastLanguageModel.from_pretrained(
      model_name=config.MODEL_NAME,
      max_seq_length=config.MAX_SEQ_LENGTH,
      dtype=None,
      load_in_4bit=True,
  )

  model = FastLanguageModel.get_peft_model(
      model,
      r=config.LORA_RANK,
      target_modules=[
          "q_proj",
          "k_proj",
          "v_proj",
          "o_proj",
          "gate_proj",
          "up_proj",
          "down_proj",
      ],
      lora_alpha=config.LORA_RANK,
      lora_dropout=0,
      bias="none",
      use_gradient_checkpointing="unsloth",
      random_state=3407,
      use_rslora=False,
      loftq_config=None,
  )

  ############################################################################
  # Train the model
  ############################################################################
  if "llama-3" in config.MODEL_NAME:
    response_template = config.PROMPT_SUFFIX.rstrip()
  else:
    response_template = config.PROMPT_SUFFIX.strip()

  collator = DataCollatorForCompletionOnlyLM(
      response_template=response_template,
      tokenizer=tokenizer)

  trainer = SFTTrainer(
      model=model,
      tokenizer=tokenizer,
      train_dataset=dataset,
      dataset_text_field="text",
      max_seq_length=config.MAX_SEQ_LENGTH,
      dataset_num_proc=2,
      packing=False,
      data_collator=collator,
      args=TrainingArguments(
          per_device_train_batch_size=16,
          gradient_accumulation_steps=1,
          warmup_steps=50,
          max_steps=config.MAX_STEPS,
          learning_rate=3e-5,
          fp16=not is_bfloat16_supported(),
          bf16=is_bfloat16_supported(),
          logging_steps=1,
          optim="adamw_8bit",
          weight_decay=0.01,
          lr_scheduler_type="linear",
          seed=3407,
          output_dir=config.MODEL_ID,
          save_steps=100,
          save_total_limit=2,
      ),
  )

  ############################################################################
  # Show current memory stats
  ############################################################################
  gpu_stats = torch.cuda.get_device_properties(0)
  start_gpu_memory = round(
      torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3
  )
  max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
  print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
  print(f"{start_gpu_memory} GB of memory reserved.")

  # Whether from scratch or resume, the following line will train the model.
  trainer_stats = trainer.train(
      resume_from_checkpoint=config.RESUME_FROM_CHECKPOINT
  )

  ############################################################################
  # Show final memory and time stats
  ############################################################################
  used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
  used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
  used_percentage = round(used_memory / max_memory * 100, 3)
  lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
  print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
  print(
      f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for"
      " training."
  )
  print(f"Peak reserved memory = {used_memory} GB.")
  print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
  print(f"Peak reserved memory % of max memory = {used_percentage} %.")
  print(
      "Peak reserved memory for training % of max memory ="
      f" {lora_percentage} %."
  )


if __name__ == "__main__":
  run_training()
