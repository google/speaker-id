"""Post-process completions for eval-ready format.

Completions are based on segmented transcripts, and may contain modified words.
We need to merge them, and use ref_to_oracle() function to oraclize it.
"""

from collections.abc import Sequence
import json

from absl import app
from absl import flags

from diarizationlm import utils


FLAGS = flags.FLAGS
flags.DEFINE_string("input", "", "Input json file.")
flags.DEFINE_string("output", "", "Output json file.")
flags.DEFINE_string("completion_suffix", "", "Suffix of the output")
flags.DEFINE_string(
    "hyp_text_field",
    "hyp_text",
    "We transfer the LLM output speakers to this text.",
)
flags.DEFINE_string(
    "hyp_spk_field",
    "hyp_spk",
    "The speakers that correspond to hyp_text_field.",
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  # Load data.
  with open(FLAGS.input, "rt") as f:
    data_dict = json.load(f)

  po = utils.PromptOptions(completion_suffix=FLAGS.completion_suffix)

  # Process utterances.
  for utt in data_dict["utterances"]:
    utils.postprocess_completions_for_utt(
        utt,
        llm_text_field="llm_text",
        llm_speaker_field="llm_spk",
        transfered_llm_speaker_field="hyp_spk_llm",
        hyp_text_field=FLAGS.hyp_text_field,
        hyp_spk_field=FLAGS.hyp_spk_field,
        po=po,
    )

  # Write output.
  with open(FLAGS.output, "wt") as f:
    json.dump(data_dict, f, indent=2)
  print("Output JSON file written to:", FLAGS.output)


if __name__ == "__main__":
  app.run(main)
