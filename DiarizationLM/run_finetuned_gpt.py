"""Run the finetuned GPT model on testing data."""

from collections.abc import Sequence
import json
from absl import app
from absl import flags
import openai
import utils

FLAGS = flags.FLAGS
flags.DEFINE_string("api_key", "", "Your OpenAI API key")
flags.DEFINE_string("engine", "", "Model id")
flags.DEFINE_string("input", "", "Input json file")
flags.DEFINE_string("output", "/tmp/output.json", "Output json file")
flags.DEFINE_integer(
    "emit_input_length", 896, "Once prompt gets larger than this, we emit"
)
flags.DEFINE_string("prompt_prefix", "", "Prefix of the input")
flags.DEFINE_string("prompt_suffix", " --> ", "Suffix of the input")
flags.DEFINE_string("completion_suffix", " [eod]", "Suffix of the input")

MAX_TOKENS = 4096


def get_completion(prompt: str) -> str:
  """Get the completion using OpenAI model."""
  completion = openai.Completion.create(
      engine=FLAGS.engine,
      prompt=prompt,
      stop=FLAGS.completion_suffix,
      max_tokens=MAX_TOKENS,
  )
  return completion.choices[0].text.strip()


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  openai.api_key = FLAGS.api_key

  # Load data.
  with open(FLAGS.input, "rt") as f:
    data_dict = json.load(f)

  po = utils.PromptOptions(
      emit_input_length=FLAGS.emit_input_length,
      prompt_prefix=FLAGS.prompt_prefix,
      prompt_suffix=FLAGS.prompt_suffix,
  )

  for utt in data_dict["utterances"]:
    prompts = utils.generate_prompts(utt, po=po)

    utt["completions"] = []
    for prompt in prompts:
      utt["completions"].append(get_completion(prompt))

  with open(FLAGS.output, "wt") as f:
    json.dump(data_dict, f, indent=2)
  print("Output JSON file written to:", FLAGS.output)


if __name__ == "__main__":
  app.run(main)
