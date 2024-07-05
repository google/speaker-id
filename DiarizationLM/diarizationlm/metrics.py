"""A Python implementation of the ASR and diarization metrics.

Note: This implementation is different from Google's internal implementation
that we used in the paper, but is a best-effort attempt to replicate the
results.
"""

import dataclasses
from typing import Any, Optional
import numpy as np
from scipy import optimize
import tqdm
from diarizationlm import utils
from diarizationlm import levenshtein


@dataclasses.dataclass
class UtteranceMetrics:
  """Metrics for one utterance."""

  wer_insert: int = 0
  wer_delete: int = 0
  wer_sub: int = 0
  wer_correct: int = 0
  wer_total: int = 0

  wder_sub: int = 0
  wder_correct: int = 0
  wder_total: int = 0


def compute_utterance_metrics(
    hyp_text: str,
    ref_text: str,
    hyp_spk: Optional[str] = None,
    ref_spk: Optional[str] = None,
) -> UtteranceMetrics:
  """Compute the word error rate of an utterance."""
  result = UtteranceMetrics()
  hyp_normalized = utils.normalize_text(hyp_text)
  ref_normalized = utils.normalize_text(ref_text)
  hyp_words = hyp_normalized.split()
  ref_words = ref_normalized.split()

  # Get the alignment.
  _, align = levenshtein.levenshtein_with_edits(ref_normalized, hyp_normalized)

  # Apply the alignment on ref speakers.
  for i, j in align:
    if i == -1:
      result.wer_insert += 1
    elif j == -1:
      result.wer_delete += 1
    else:
      if ref_words[i] == hyp_words[j]:
        result.wer_correct += 1
      else:
        result.wer_sub += 1

  result.wer_total = result.wer_correct + result.wer_sub + result.wer_delete
  assert result.wer_total == len(ref_words)

  compute_wder = hyp_spk or ref_spk
  if not compute_wder:
    return result

  if not (hyp_spk and ref_spk):
    raise ValueError("hyp_spk and ref_spk must be both unset or both set.")

  hyp_spk_list = [int(x) for x in hyp_spk.split()]
  ref_spk_list = [int(x) for x in ref_spk.split()]
  if len(hyp_spk_list) != len(hyp_words):
    raise ValueError("hyp_spk and hyp_text must have the same length.")
  if len(ref_spk_list) != len(ref_words):
    raise ValueError("ref_spk and ref_text must have the same length.")
  hyp_spk_list_align = []
  ref_spk_list_align = []

  for i, j in align:
    if i != -1 and j != -1:
      ref_spk_list_align.append(ref_spk_list[i])
      hyp_spk_list_align.append(hyp_spk_list[j])

  # Build cost matrix.
  max_spk = max(max(ref_spk_list_align), max(hyp_spk_list_align))
  cost_matrix = np.zeros((max_spk, max_spk), dtype=int)
  for aligned, original in zip(ref_spk_list_align, hyp_spk_list_align):
    cost_matrix[aligned - 1, original - 1] += 1

  # Solve alignment.
  row_index, col_index = optimize.linear_sum_assignment(
      cost_matrix, maximize=True
  )
  result.wder_correct = int(cost_matrix[row_index, col_index].sum())
  result.wder_total = len(ref_spk_list_align)
  result.wder_sub = result.wder_total - result.wder_correct

  return result


def compute_metrics_on_json_dict(
    json_dict: dict[str, Any],
    ref_text_field: str = "ref_text",
    hyp_text_field: str = "hyp_text",
    ref_spk_field: str = "ref_spk",
    hyp_spk_field: str = "hyp_spk",
) -> dict[str, Any]:
  """Compute metrics for all utterances in a json object."""
  result_dict = {
      "utterances": [],
  }
  for utt in tqdm.tqdm(json_dict["utterances"]):
    utt_metrics = compute_utterance_metrics(
        hyp_text=utt[hyp_text_field],
        ref_text=utt[ref_text_field],
        hyp_spk=utt[hyp_spk_field],
        ref_spk=utt[ref_spk_field],
    )
    utt_result = dataclasses.asdict(utt_metrics)
    utt_result["utterance_id"] = utt["utterance_id"]
    result_dict["utterances"].append(utt_result)

  final_wer_total = 0
  final_wer_correct = 0
  final_wer_sub = 0
  final_wer_delete = 0
  final_wer_insert = 0
  final_wder_total = 0
  final_wder_correct = 0
  final_wder_sub = 0
  for utt in result_dict["utterances"]:
    final_wer_total += utt["wer_total"]
    final_wer_correct += utt["wer_correct"]
    final_wer_sub += utt["wer_sub"]
    final_wer_delete += utt["wer_delete"]
    final_wer_insert += utt["wer_insert"]
    final_wder_total += utt["wder_total"]
    final_wder_correct += utt["wder_correct"]
    final_wder_sub += utt["wder_sub"]

  final_wer = (
      final_wer_sub + final_wer_delete + final_wer_insert
  ) / final_wer_total
  final_wder = final_wder_sub / final_wder_total
  result_dict["WER"] = final_wer
  result_dict["WDER"] = final_wder
  return result_dict
