"""Compute metrics on a json file of utterances."""

from collections.abc import Sequence
import json

from absl import app
from absl import flags

from diarizationlm import metrics


FLAGS = flags.FLAGS
flags.DEFINE_string("input", "", "Input json file of utterances.")
flags.DEFINE_string(
    "output",
    "/tmp/metrics.json",
    "Output json file of metrics.",
)
flags.DEFINE_string(
    "ref_text_field",
    "ref_text",
    "The field name of the reference text in the input json file.",
)
flags.DEFINE_string(
    "hyp_text_field",
    "hyp_text",
    "The field name of the hypothesis text in the input json file.",
)
flags.DEFINE_string(
    "ref_spk_field",
    "ref_spk",
    "The field name of the reference speakers in the input json file.",
)
flags.DEFINE_string(
    "hyp_spk_field",
    "hyp_spk",
    "The field name of the hypothesis speakers in the input json file.",
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  with open(FLAGS.input) as f:
    json_dict = json.load(f)
  result_dict = metrics.compute_metrics_on_json_dict(
      json_dict,
      ref_text_field=FLAGS.ref_text_field,
      hyp_text_field=FLAGS.hyp_text_field,
      ref_spk_field=FLAGS.ref_spk_field,
      hyp_spk_field=FLAGS.hyp_spk_field,
  )
  with open(FLAGS.output, "wt") as f:
    json.dump(result_dict, f, indent=2)
  print("Output JSON file written to:", FLAGS.output)


if __name__ == "__main__":
  app.run(main)
