"""Run the finetuned GPT model on testing data."""

import json
import os

import config
import tqdm
from diarizationlm import utils
from unsloth import FastLanguageModel


def get_completion(prompt: str, model, tokenizer) -> str:
  """Get the completion using OpenAI model."""
  inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

  outputs = model.generate(
      **inputs, max_new_tokens=inputs.input_ids.shape[1] * 1.2, use_cache=True
  )
  completion = tokenizer.batch_decode(
      outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
  )
  return completion[0]


def run_inference(input_file: str, output_dir: str):
  print("Running inference on:", input_file)

  # Output dir.
  print("Output directory:", output_dir)
  os.makedirs(output_dir, exist_ok=True)

  # Load model.
  model, tokenizer = FastLanguageModel.from_pretrained(
      model_name=os.path.join(config.MODEL_ID, "model"),
      max_seq_length=config.MAX_SEQ_LENGTH,
      dtype=None,
  )
  FastLanguageModel.for_inference(model)

  # Load data.
  with open(input_file, "rt") as f:
    data_dict = json.load(f)

  po = utils.PromptOptions(
      emit_input_length=config.EMIT_INPUT_LENGTH,
      prompt_prefix=config.PROMPT_PREFIX,
      prompt_suffix=config.PROMPT_SUFFIX,
  )

  for utt in tqdm.tqdm(data_dict["utterances"]):
    prompts = utils.generate_prompts(utt, po=po)

    utt["completions"] = []
    for prompt in prompts:
      utt["completions"].append(get_completion(prompt, model, tokenizer))

  with open(os.path.join(output_dir, "final.json"), "wt") as f:
    json.dump(data_dict, f, indent=2)
  print("Final output JSON file written to:", output_dir)


if __name__ == "__main__":
  for eval_dataset in config.EVAL_INPUTS:
    print("Running inference on:", eval_dataset)
    eval_input = config.EVAL_INPUTS[eval_dataset]
    output_dir = os.path.join(
        config.MODEL_ID,
        "decoded",
        f"checkpoint-{config.CHECKPOINT}",
        eval_dataset,
    )
    run_inference(input_file=eval_input, output_dir=output_dir)
