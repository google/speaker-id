"""Post-process completions and compute metrics."""
import os
import json
import tqdm

import config

import colortimelog
from diarizationlm import utils
from diarizationlm import metrics


def postprocess(input_file: str, output_file: str) -> None:
  # Load data.
  with open(input_file, "rt") as f:
    data_dict = json.load(f)

  po = utils.PromptOptions(completion_suffix=config.COMPLETION_SUFFIX)

  # Process utterances.
  for utt in tqdm.tqdm(data_dict["utterances"]):
    utils.postprocess_completions_for_utt(
        utt,
        llm_text_field="llm_text",
        llm_speaker_field="llm_spk",
        transfered_llm_speaker_field="hyp_spk_llm",
        hyp_text_field="hyp_text",
        hyp_spk_field="hyp_spk",
        po=po,
    )

  # Write output.
  with open(output_file, "wt") as f:
    json.dump(data_dict, f, indent=2)
  print("Output JSON processed file written to:", output_file)


def evaluate(input_file: str, output_file: str) -> None:
  with open(input_file) as f:
    json_dict = json.load(f)
  result_dict = metrics.compute_metrics_on_json_dict(
      json_dict,
      ref_text_field="ref_text",
      hyp_text_field="hyp_text",
      ref_spk_field="ref_spk",
      hyp_spk_field="hyp_spk_llm",
  )
  with open(output_file, "wt") as f:
    json.dump(result_dict, f, indent=2)
  print("Output JSON metrics file written to:", output_file)


if __name__ == "__main__":
  for eval_dataset in config.EVAL_INPUTS:
    with colortimelog.timeblock("Evaluating: " + eval_dataset):
      output_dir = os.path.join(config.MODEL_ID,
                                "decoded",
                                f"checkpoint-{config.CHECKPOINT}",
                                eval_dataset)
      postprocess(
        input_file=os.path.join(output_dir, "final.json"),
        output_file=os.path.join(output_dir, "postprocessed.json"))
      evaluate(
        input_file=os.path.join(output_dir, "postprocessed.json"),
        output_file=os.path.join(output_dir, "metrics.json"))
