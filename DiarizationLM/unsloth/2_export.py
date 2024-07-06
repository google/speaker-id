"""Export the model."""

import os
import config
from unsloth import FastLanguageModel


def export_models(
    save_lora: bool = True,
    save_16bit: bool = True,
) -> None:
  ############################################################################
  # Get model
  ############################################################################
  checkpoint_path = os.path.join(
      config.MODEL_ID, f"checkpoint-{config.CHECKPOINT}"
  )
  print(f"Loading model from {checkpoint_path}...")
  model, tokenizer = FastLanguageModel.from_pretrained(
      model_name=checkpoint_path,
      max_seq_length=config.MAX_SEQ_LENGTH,
      dtype=None,
      load_in_4bit=True,
  )

  if save_lora:
    print("Saving LoRA model...")
    model.save_pretrained(
        os.path.join(config.MODEL_ID, "lora_model")
    )  # Local saving
    tokenizer.save_pretrained(os.path.join(config.MODEL_ID, "lora_model"))

  if save_16bit:
    print("Saving 16bit model...")
    model.save_pretrained_merged(
        os.path.join(config.MODEL_ID, "model"),
        tokenizer,
        save_method="merged_16bit",
    )


if __name__ == "__main__":
  export_models()
