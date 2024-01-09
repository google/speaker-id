"""Prepare training data for LLM."""
from collections.abc import Sequence

from absl import app
from absl import flags
import json
import tensorflow as tf
import utils


FLAGS = flags.FLAGS
flags.DEFINE_string("input", "", "Comma-separated list of json files")
flags.DEFINE_string("output", "/tmp/output", "Output file.")
flags.DEFINE_enum(
    "output_type",
    "tfrecord",
    ["tfrecord", "json", "csv"],
    "Output container formats for different use cases.",
)
flags.DEFINE_string("text_field", "hyp_text", "Name of field to get text")
flags.DEFINE_string(
    "input_speaker_field", "hyp_spk", "Name of field to get input speakers"
)
flags.DEFINE_string(
    "target_speaker_field",
    "hyp_spk_oracle",
    "Name of field to get output speakers",
)
flags.DEFINE_integer(
    "emit_input_length", 896, "Max length of prompt"
)
flags.DEFINE_integer(
    "emit_target_length", 896, "Max length of target (completion)"
)
flags.DEFINE_string("prompt_prefix", "", "Prefix of the input")
flags.DEFINE_string("prompt_suffix", " --> ", "Suffix of the input")
flags.DEFINE_string("completion_suffix", "", "Suffix of the output")
flags.DEFINE_string(
    "input_feature_key",
    "inputs",
    "This is the input feature key for the LLM prompt in the output type.",
)
flags.DEFINE_string(
    "output_feature_key",
    "targets",
    "This is the output feature key for the LLM completion in the output type.",
)


def main(argv: Sequence[str]) -> None:
  del argv
  po = utils.PromptOptions(
      emit_input_length=FLAGS.emit_input_length,
      emit_target_length=FLAGS.emit_target_length,
      prompt_prefix=FLAGS.prompt_prefix,
      prompt_suffix=FLAGS.prompt_suffix,
      completion_suffix=FLAGS.completion_suffix,
  )

  reader = utils.JsonUtteranceReader(
      json_files=FLAGS.input,
      text_field=FLAGS.text_field,
      input_speaker_field=FLAGS.input_speaker_field,
      target_speaker_field=FLAGS.target_speaker_field,
      po=po,
  )

  if FLAGS.output_type == "tfrecord":
    with tf.io.TFRecordWriter(FLAGS.output) as writer:
      for _, prompt, target in reader.generate_data_tuple():
        example = tf.train.Example()
        example.features.feature[
            FLAGS.input_feature_key
        ].bytes_list.value.append(prompt.encode("utf-8"))
        example.features.feature[
            FLAGS.output_feature_key
        ].bytes_list.value.append(target.encode("utf-8"))
        writer.write(example.SerializeToString())
  elif FLAGS.output_type == "json":
    output_dict = {"utterances": []}
    for key, prompt, target in reader.generate_data_tuple():
      segment = dict()
      segment["utterance_id"] = key
      segment[FLAGS.input_feature_key] = prompt
      segment[FLAGS.output_feature_key] = target
      output_dict["utterances"].append(segment)
    with open(FLAGS.output, "wt") as f:
      json.dump(output_dict, f, indent=2)
  elif FLAGS.output_type == "csv":
    # For CSV header, we normally use "prompt,completion".
    csv_lines = [FLAGS.input_feature_key + "," + FLAGS.output_feature_key]
    for _, prompt, target in reader.generate_data_tuple():
      csv_lines.append('"{}","{}"'.format(prompt, target))
    with open(FLAGS.output, "wt") as f:
      f.write("\n".join(csv_lines))

  print("Output has been written to:", FLAGS.output)


if __name__ == "__main__":
  app.run(main)
