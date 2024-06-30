import config
from datasets import Dataset, disable_caching, concatenate_datasets
from diarizationlm import utils


def formatting_prompts_func(example):
  return {"text": example["prompt"] + example["target"]}


def build_dataset_single_source(input_file: str):
  disable_caching()
  po = utils.PromptOptions(
      emit_input_length=config.EMIT_INPUT_LENGTH,
      emit_target_length=config.EMIT_TARGET_LENGTH,
      prompt_prefix=config.PROMPT_PREFIX,
      prompt_suffix=config.PROMPT_SUFFIX,
      completion_suffix=config.COMPLETION_SUFFIX,
  )

  reader_hyp2ora = utils.JsonUtteranceReader(
      json_files=input_file,
      text_field="hyp_text",
      input_speaker_field="hyp_spk",
      target_speaker_field="hyp_spk_oracle",
      po=po,
  )
  reader_deg2ref = utils.JsonUtteranceReader(
      json_files=input_file,
      text_field="ref_text",
      input_speaker_field="ref_spk_degraded",
      target_speaker_field="ref_spk",
      po=po,
  )
  dataset1 = Dataset.from_generator(reader_hyp2ora.generate_data_dict)
  dataset2 = Dataset.from_generator(reader_deg2ref.generate_data_dict)
  return concatenate_datasets([dataset1, dataset2])


def build_dataset():
  disable_caching()
  all_datasets = []
  for data_name in config.TRAINING_INPUT:
    data_path, data_repeat = config.TRAINING_INPUT[data_name]
    all_datasets.extend([build_dataset_single_source(data_path)] * data_repeat)
  dataset = concatenate_datasets(all_datasets)
  dataset = dataset.shuffle(seed=42)
  dataset = dataset.map(formatting_prompts_func)
  return dataset
