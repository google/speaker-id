"""Export the model."""

import os
import config
from unsloth import FastLanguageModel


def export_models(
    save_lora=True,
    save_16bit=True,
    save_4bit=False,
    save_gguf=False,
    save_4bit_gguf=True,
):
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

  if save_4bit:
    # Note: This model current has problems.
    # Error: "does not contain `bitsandbytes__*` and possibly other
    # `quantized_stats` components."
    print("Saving 4bit model...")
    model.save_pretrained_merged(
        os.path.join(config.MODEL_ID, "4bit_model"),
        tokenizer,
        save_method="merged_4bit_forced",
    )

  if save_gguf:
    print("Saving GGUF model...")
    model.save_pretrained_gguf(
        os.path.join(config.MODEL_ID, "model"),
        tokenizer,
        quantization_method="f16",
    )

  if save_4bit_gguf:
    print("Saving 4-bit GGUF model...")
    model.save_pretrained_gguf(
        os.path.join(config.MODEL_ID, "model"),
        tokenizer,
        quantization_method="q4_k_m",
    )


if __name__ == "__main__":
  export_models()
