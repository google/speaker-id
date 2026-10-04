"""Multi-domain dataset builder with Locality-Preserving Oracle curation."""

import json
import config_gemma4_e4b as config
from datasets import Dataset, concatenate_datasets, disable_caching
from diarizationlm import utils


def formatting_prompts_func(example: dict[str, str]) -> dict[str, str]:
  return {"text": example["prompt"] + example["target"]}


def _generate_locality_data_dicts(
    input_file: str,
    text_field: str,
    input_speaker_field: str,
    target_speaker_field: str,
    po: utils.PromptOptions,
    max_local_span_words: int = 5,
):
  """Yields prompt-target dicts with locality-preserving oracle speakers."""
  with open(input_file, "rt") as f:
    data_dict = json.load(f)

  for utt in data_dict["utterances"]:
    utt_copy = dict(utt)
    utt_copy[target_speaker_field] = (
        utils.get_locality_preserving_oracle_speakers(
            hyp_spk=utt_copy[input_speaker_field],
            hyp_spk_oracle=utt_copy[target_speaker_field],
            max_local_span_words=max_local_span_words,
        )
    )
    reader = utils.JsonUtteranceReader(
        json_files="",
        text_field=text_field,
        input_speaker_field=input_speaker_field,
        target_speaker_field=target_speaker_field,
        po=po,
        utt=utt_copy,
    )
    yield from reader.generate_data_dict()


def build_dataset_single_source(input_file: str) -> Dataset:
  """Builds a single-source dataset with Locality-Preserving Oracle targets."""
  disable_caching()
  po = utils.PromptOptions(
      emit_input_length=config.EMIT_INPUT_LENGTH,
      emit_target_length=config.EMIT_TARGET_LENGTH,
      prompt_prefix=config.PROMPT_PREFIX,
      prompt_suffix=config.PROMPT_SUFFIX,
      completion_suffix=config.COMPLETION_SUFFIX,
  )
  max_local_span_words = getattr(config, "MAX_LOCAL_SPAN_WORDS", 5)

  dataset1 = Dataset.from_generator(
      lambda: _generate_locality_data_dicts(
          input_file=input_file,
          text_field="hyp_text",
          input_speaker_field="hyp_spk",
          target_speaker_field="hyp_spk_oracle",
          po=po,
          max_local_span_words=max_local_span_words,
      )
  )
  dataset2 = Dataset.from_generator(
      lambda: _generate_locality_data_dicts(
          input_file=input_file,
          text_field="ref_text",
          input_speaker_field="ref_spk_degraded",
          target_speaker_field="ref_spk",
          po=po,
          max_local_span_words=max_local_span_words,
      )
  )
  return concatenate_datasets([dataset1, dataset2])


def build_dataset() -> Dataset:
  """Builds the full multi-domain training dataset."""
  disable_caching()
  all_datasets = []
  for data_name in config.TRAINING_INPUT:
    data_path, data_repeat = config.TRAINING_INPUT[data_name]
    all_datasets.extend([build_dataset_single_source(data_path)] * data_repeat)
  dataset = concatenate_datasets(all_datasets)
  dataset = dataset.shuffle(seed=42)
  dataset = dataset.map(formatting_prompts_func)
  return dataset
