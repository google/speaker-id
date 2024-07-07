"""Utility functions and classes."""
from collections.abc import Generator, Sequence
import copy
import dataclasses
import json
import sys
from typing import Any, List, Optional

import numpy as np
from scipy import optimize

from diarizationlm import levenshtein

PUNCTUATIONS = [",", ".", "_", "?", "!", "-", '"', "'"]


@dataclasses.dataclass
class PromptOptions:
  """Options for generating prompts."""

  # For prompt segmentation.
  emit_input_length: int = 896
  emit_target_length: int = 896

  # Prefix and suffix for prompt and completion.
  # As a reference, OpenAI finetuning API usually suggests:
  # - No prompt prefix
  # - Prompt suffix: " -> "
  # - Completion suffix: " END"
  prompt_prefix: str = ""
  prompt_suffix: str = " --> "
  completion_suffix: str = ""

  # How do we represent the speaker token.
  # We may consider shorter prefix for token efficiency.
  speaker_prefix: str = "<speaker:"
  speaker_suffix: str = ">"


def normalize_text(text: str) -> str:
  """Normalize text."""
  # Convert to lower case.
  text_lower = text.lower().strip()

  # Remove punctuation.
  for punc in PUNCTUATIONS:
    text_lower = text_lower.replace(punc, "")

  return " ".join(text_lower.split())


def speakers_transform(speakers: Sequence[str]) -> list[str]:
  """Transform list of speakers to be order based."""
  spk_map = {}
  index = 0
  for spk in speakers:
    if spk not in spk_map:
      index += 1
      spk_map[spk] = index
  return [str(spk_map[spk]) for spk in speakers]


def get_aligned_hyp_speakers(
    hyp_text: str,
    ref_text: str,
    ref_spk: str,
    print_debug_info: bool = False,
) -> str:
  """Align ref_text to hyp_text, then apply the alignment to ref_spk."""
  # Counters for insertions and deletions in hyp and ref text alignment.
  num_insertions, num_deletions = 0, 0

  # Get the alignment.
  _, align = levenshtein.levenshtein_with_edits(
      normalize_text(ref_text), normalize_text(hyp_text)
  )

  ref_spk_list = ref_spk.split()
  hyp_spk_align = []

  # Apply the alignment on ref speakers.
  for i, j in align:
    if i == -1:
      # hyp has insertion
      hyp_spk_align.append("-1")
      num_insertions += 1
    elif j == -1:
      # hyp has deletion
      num_deletions += 1
      continue
    else:
      hyp_spk_align.append(ref_spk_list[i])
  hyp_spk_align = " ".join(hyp_spk_align)

  if print_debug_info:
    print("Number of insertions: ", num_insertions)
    print("Number of deletions: ", num_deletions)
    # This is not the traditional denominator of WER. Instead, this is
    # len(hyp) + len(ref) - len(SUB).
    print("Length of align pairs: ", len(align))
  return hyp_spk_align


def get_oracle_speakers(hyp_spk: str, hyp_spk_align: str) -> Sequence[int]:
  """Get the oracle speakers for hypothesis."""
  hyp_spk_list = [int(x) for x in hyp_spk.split()]
  hyp_spk_align_list = [int(x) for x in hyp_spk_align.split()]

  # Build cost matrix.
  max_spk = max(max(hyp_spk_list), max(hyp_spk_align_list))
  cost_matrix = np.zeros((max_spk, max_spk))
  for aligned, original in zip(hyp_spk_align_list, hyp_spk_list):
    cost_matrix[aligned - 1, original - 1] += 1

  # Solve alignment.
  row_index, col_index = optimize.linear_sum_assignment(
      cost_matrix, maximize=True
  )

  # Build oracle.
  hyp_spk_oracle = hyp_spk_list.copy()
  for i in range(len(hyp_spk_list)):
    if hyp_spk_align_list[i] == -1:
      # There are some missing words. In such cases, we just use the original
      # speaker for these words if possible.
      if hyp_spk_list[i] == -1:
        # If we don't have original speaker for missing words, just use the
        # previous speaker if possible.
        # This is useful for the update_hyp_text_in_utt_dict() function.
        if i == 0:
          hyp_spk_oracle[i] = 1
        else:
          hyp_spk_oracle[i] = hyp_spk_oracle[i - 1]
      continue
    assert row_index[hyp_spk_align_list[i] - 1] == hyp_spk_align_list[i] - 1
    hyp_spk_oracle[i] = col_index[hyp_spk_align_list[i] - 1] + 1

  return hyp_spk_oracle


# Transcript-Preserving Speaker Transfer (TPST)
def transcript_preserving_speaker_transfer(
    src_text: str, src_spk: str, tgt_text: str, tgt_spk: str
) -> str:
  """Apply source speakers to target."""
  if len(tgt_text.split()) != len(tgt_spk.split()):
    raise ValueError("tgt_text and tgt_spk must have the same length")
  if len(src_text.split()) != len(src_spk.split()):
    raise ValueError("src_text and src_spk must have the same length")
  tgt_spk_align = get_aligned_hyp_speakers(
      hyp_text=tgt_text,
      ref_text=src_text,
      ref_spk=src_spk,
  )
  oracle_speakers = get_oracle_speakers(
      hyp_spk=tgt_spk, hyp_spk_align=tgt_spk_align
  )
  return " ".join([str(x) for x in oracle_speakers])


# We can use this to finetune LLM.
# Inputs (prompts): hyp diarized text
# Targets: hyp diarized text with oracle speakers
def ref_to_oracle(json_dict: dict[str, str]) -> str:
  """Apply reference speakers to hypothesis."""
  return transcript_preserving_speaker_transfer(
      src_text=json_dict["ref_text"],
      src_spk=json_dict["ref_spk"],
      tgt_text=json_dict["hyp_text"],
      tgt_spk=json_dict["hyp_spk"],
  )


# Similar to ref_to_oracle, but the opposite direction.
# We can use this to finetune LLM.
# Inputs (prompts): ref diarized text with degraded speakers
# Targets: ref diarized text
def hyp_to_degraded(json_dict: dict[str, str]) -> str:
  """Apply hypothesis speakers to reference."""
  return transcript_preserving_speaker_transfer(
      src_text=json_dict["hyp_text"],
      src_spk=json_dict["hyp_spk"],
      tgt_text=json_dict["ref_text"],
      tgt_spk=json_dict["ref_spk"],
  )


def create_diarized_text(
    word_labels: Sequence[str],
    speaker_labels: Sequence[str],
    use_new_line: bool = False,
    po: PromptOptions = PromptOptions(),
) -> str:
  """Create diarized text from words and speaker labels."""
  output = []
  previous_speaker = None
  for word, speaker in zip(word_labels, speaker_labels):
    if speaker != previous_speaker:
      if previous_speaker and use_new_line:
        output.append("\n")
      output.append(po.speaker_prefix + speaker + po.speaker_suffix)
    output.append(word)
    previous_speaker = speaker
  return " ".join(output)


def extract_text_and_spk(
    completions: str, po: PromptOptions, skip_meaningless_speaker: bool = True
) -> tuple[str, str]:
  """Extract the text and spk from the completions string."""
  spk = "1"
  previous_spk = "1"
  result_text = []
  result_spk = []
  for word in completions.split():
    if word.startswith(po.speaker_prefix):
      if not word.endswith(po.speaker_suffix):
        word += po.speaker_suffix
      spk = word[len(po.speaker_prefix):-len(po.speaker_suffix)]
      # Handle undefined behaviors of non-recognizable spk with a placeholder.
      try:
        spk_int = int(spk)
        if not spk or spk_int < 1 or spk_int > 10:
          raise ValueError("Seeing unexpected word: ", word)
        previous_spk = spk
      except ValueError as exc:
        if skip_meaningless_speaker:
          print("Skipping meaningless speaker token:", word)
          spk = previous_spk
        else:
          raise exc
    else:
      result_text.append(word)
      result_spk.append(spk)
  return " ".join(result_text), " ".join(result_spk)


def discard_empty_str_and_remove_boundary_white_space(
    inputs: List[str],
) -> List[str]:
  return [x.strip() for x in inputs if x.strip()]


@dataclasses.dataclass
class JsonUtteranceReader:
  """Read the json files and generate prompts and targets."""

  json_files: str  # Ignored if utt is given.
  text_field: str
  input_speaker_field: str
  target_speaker_field: str  # If not given, will skip targets.
  po: PromptOptions
  utt: dict[str, str] = dataclasses.field(default_factory=dict)

  def generate_utts(self) -> Generator[dict[str, str], None, None]:
    """Generate an utterance from all json files."""
    if self.utt:
      yield self.utt
      return

    for json_file in self.json_files.split(","):
      with open(json_file) as f:
        data_dict = json.load(f)
        for utt in data_dict["utterances"]:
          yield utt

  def generate_data_tuple(self) -> Generator[tuple[str, str, str], None, None]:
    """Generate uttid-prompt-target tuples."""
    for utt in self.generate_utts():
      yield from self.generate_data_tuple_for_utt(utt)

  def generate_data_dict(self) -> Generator[dict[str, str], None, None]:
    """Generate a dict that can be used for datasets.Dataset.from_generator."""
    for uttid, prompt, target in self.generate_data_tuple():
      yield {"uttid": uttid, "prompt": prompt, "target": target}

  def generate_data_tuple_for_utt(
      self, utt: dict[str, str]
  ) -> Generator[tuple[str, str, str], None, None]:
    """Generate uttid-prompt-target tuples from a single utterance."""
    self.seg_id = 0
    utt_id = utt["utterance_id"]

    # Get the fields from the utterance.
    words = discard_empty_str_and_remove_boundary_white_space(
        utt[self.text_field].split(" ")
    )
    p_speakers = discard_empty_str_and_remove_boundary_white_space(
        utt[self.input_speaker_field].split(" ")
    )
    assert len(words) == len(p_speakers)
    if self.target_speaker_field:
      t_speakers = discard_empty_str_and_remove_boundary_white_space(
          utt[self.target_speaker_field].split(" ")
      )
      assert len(words) == len(t_speakers)
    else:
      t_speakers = []

    yield from self.generate_data_tuple_from_range(
        utt_id, words, p_speakers, t_speakers, start=0, end=len(words)
    )

  def generate_data_tuple_from_range(
      self, utt_id, words, p_speakers, t_speakers, start, end
  ) -> Generator[tuple[str, str, str], None, None]:
    """Generate uttid-prompt-target tuples from a range of words."""
    # Decide whether to call recursively from the estimated length.
    estimated_prompt_length = (
        len(self.po.prompt_prefix)
        + len(" ".join(words[start:end]))
        + len(self.po.prompt_suffix)
    )
    if (
        estimated_prompt_length > self.po.emit_input_length
        or estimated_prompt_length > self.po.emit_target_length
    ):
      yield from self.generate_data_tuple_from_range(
          utt_id, words, p_speakers, t_speakers, start, (start + end) // 2
      )
      yield from self.generate_data_tuple_from_range(
          utt_id, words, p_speakers, t_speakers, (start + end) // 2, end
      )
      return

    prompt = self.po.prompt_prefix
    previous_p_spk = ""
    target = ""
    previous_t_spk = ""

    # Main loop.
    for i in range(start, end):
      word = words[i]
      p_spk = p_speakers[i]
      if p_spk != previous_p_spk:
        if previous_p_spk:
          prompt += " "
        prompt += self.po.speaker_prefix + p_spk + self.po.speaker_suffix
      prompt += " " + word
      previous_p_spk = p_spk

      if self.target_speaker_field:
        t_spk = t_speakers[i]
        if t_spk != previous_t_spk:
          if previous_t_spk:
            target += " "
          target += self.po.speaker_prefix + t_spk + self.po.speaker_suffix
        target += " " + word
        previous_t_spk = t_spk

    prompt_id = utt_id + "_seg" + str(self.seg_id)
    prompt += self.po.prompt_suffix
    target += self.po.completion_suffix
    if (
        len(prompt) <= self.po.emit_input_length
        and len(target) <= self.po.emit_target_length
    ):
      yield (prompt_id, prompt, target)
      self.seg_id += 1
    else:
      yield from self.generate_data_tuple_from_range(
          utt_id, words, p_speakers, t_speakers, start, (start + end) // 2
      )
      yield from self.generate_data_tuple_from_range(
          utt_id, words, p_speakers, t_speakers, (start + end) // 2, end
      )


def generate_prompts(
    utt: dict[str, str],
    po: PromptOptions,
    text_field: str = "hyp_text",
    input_speaker_field: str = "hyp_spk",
) -> list[str]:
  """Generate a list of prompts for a given utt."""
  po_modified = copy.deepcopy(po)
  po_modified.emit_target_length = sys.maxsize
  reader = JsonUtteranceReader(
      json_files="",
      text_field=text_field,
      input_speaker_field=input_speaker_field,
      target_speaker_field="",
      po=po_modified,
      utt=utt,
  )
  prompts = []
  for _, prompt, _ in reader.generate_data_tuple():
    prompts.append(prompt)
  if len(prompts) > 1:
    for prompt in prompts:
      if len(prompt) < po.emit_input_length / 3:
        raise RuntimeError("Prompt too short: ", prompt)
  return prompts


def find_utt_dict(
    utt_id: str, data_dict: dict[str, Any]
) -> Optional[dict[str, str]]:
  """Find a utt_dict with a speicifc utterance_id from data_dict."""
  for utt_dict in data_dict["utterances"]:
    if utt_dict["utterance_id"] == utt_id:
      return utt_dict
  return None


def update_hyp_text_in_utt_dict(
    input_utt_dict: dict[str, str], new_hyp_text
) -> dict[str, str]:
  """Update the hyp_text of a json utt_dict.

  We also transfer its original hyp_spk to the new hyp_text.

  This is useful if we want to use USM ASR transcripts to replace the
  turn-to-diarize transcripts, as the WER of turn-to-diarize transcripts is too
  high.

  Args:
    input_utt_dict: the input utt_dict
    new_hyp_text: the new hyp_text

  Returns:
    the new utt_dict
  """
  utt_dict = copy.deepcopy(input_utt_dict)
  # We don't know the speakers for new_hyp_text, so just use -1 as initial
  # speakers.
  new_hyp_spk = transcript_preserving_speaker_transfer(
      src_text=utt_dict["hyp_text"],
      src_spk=utt_dict["hyp_spk"],
      tgt_text=new_hyp_text,
      tgt_spk=" ".join(["-1" for _ in new_hyp_text.split()]),
  )
  # Update the utt_dict.
  utt_dict["hyp_text"] = new_hyp_text
  utt_dict["hyp_spk"] = new_hyp_spk
  utt_dict["hyp_diarized_text"] = create_diarized_text(
      new_hyp_text.split(), new_hyp_spk.split()
  )
  return utt_dict


def truncate_suffix_and_tailing_text(text: str, suffix: str) -> str:
  """Tailing text after suffix should be removed as well."""
  if suffix and suffix in text:
    return text[: text.find(suffix)]
  return text


def postprocess_completions_for_utt(
    utt: dict[str, Any],
    llm_text_field: str = "llm_text",
    llm_speaker_field: str = "llm_spk",
    transfered_llm_speaker_field: str = "hyp_spk_llm",
    hyp_text_field: str = "hyp_text",
    hyp_spk_field: str = "hyp_spk",
    po: PromptOptions = PromptOptions(),
) -> None:
  """Postprocess the LLM completions of an utterance json dict."""
  # Remove completion suffix if it exists.
  completions_list = []
  for completion in utt["completions"]:
    if po.completion_suffix and po.completion_suffix in completion:
      completion = truncate_suffix_and_tailing_text(
          completion, po.completion_suffix
      )
    completions_list.append(completion)
  completions = " ".join(completions_list).strip()

  # Extract text and speaker.
  utt[llm_text_field], utt[llm_speaker_field] = extract_text_and_spk(
      completions, po=po
  )
  # Tha TPST alignment on LLM output against recognized hypothesis text can be
  # considered as a postprocessing step to ensure the hypothesis text does not
  # change too much from the diarization baseline.
  # Note: this step can arguably be skipped and we directly use LLM output
  # for evaluation. The assumption is LLM does not change original text too
  # much. `update_sstable_with_speakers` should be updated accordingly if so.
  utt[transfered_llm_speaker_field] = transcript_preserving_speaker_transfer(
      src_text=utt[llm_text_field],
      src_spk=utt[llm_speaker_field],
      tgt_text=utt[hyp_text_field],
      tgt_spk=utt[hyp_spk_field],
  )


def transfer_llm_completion(
    llm_completion: str,
    hyp: str,
    po: PromptOptions = PromptOptions(),
) -> str:
  """Transfer the LLM completion text to use text from hypothesis."""
  llm_text, llm_speaker = extract_text_and_spk(
      llm_completion, po=po
  )
  hyp_text, hyp_speaker = extract_text_and_spk(
      hyp, po=po
  )
  transfered_llm_speaker = transcript_preserving_speaker_transfer(
      src_text=llm_text,
      src_spk=llm_speaker,
      tgt_text=hyp_text,
      tgt_spk=hyp_speaker,
  )
  transferred = create_diarized_text(
      word_labels=hyp_text.split(),
      speaker_labels=transfered_llm_speaker.split(),
      po=po,
  )
  return transferred
