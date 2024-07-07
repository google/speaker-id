"""Tests for metrics."""

import json
import os
from diarizationlm import metrics
import unittest


class MetricsTest(unittest.TestCase):

  def test_wer_same_utt(self):
    hyp = "hello good morning how are you"
    ref = "Hello. Good morning, how are you?"
    result = metrics.compute_utterance_metrics(hyp, ref)
    self.assertEqual(result.wer_insert, 0)
    self.assertEqual(result.wer_delete, 0)
    self.assertEqual(result.wer_sub, 0)
    self.assertEqual(result.wer_correct, 6)
    self.assertEqual(result.wer_total, 6)

  def test_wer_diff_utt(self):
    hyp = "hi morning how are you doing sir"
    ref = "Hello. Good morning, how are you"
    result = metrics.compute_utterance_metrics(hyp, ref)
    self.assertEqual(result.wer_insert, 2)
    self.assertEqual(result.wer_delete, 1)
    self.assertEqual(result.wer_sub, 1)
    self.assertEqual(result.wer_correct, 4)
    self.assertEqual(result.wer_total, 6)

  def test_wer_special(self):
    ref = "a b c"
    hyp = "x y a c"
    result = metrics.compute_utterance_metrics(hyp, ref)
    self.assertEqual(result.wer_insert, 1)
    self.assertEqual(result.wer_delete, 0)
    self.assertEqual(result.wer_sub, 2)
    self.assertEqual(result.wer_correct, 1)
    self.assertEqual(result.wer_total, 3)

  def test_wder_same_words(self):
    hyp = "hello good morning how are you"
    ref = "Hello. Good morning, how are you?"
    hyp_spk = "1 1 1 2 2 2"
    ref_spk = "2 2 2 2 1 1"
    result = metrics.compute_utterance_metrics(hyp, ref, hyp_spk, ref_spk)
    self.assertEqual(result.wer_correct, 6)
    self.assertEqual(result.wder_sub, 1)
    self.assertEqual(result.wder_correct, 5)
    self.assertEqual(result.wder_total, 6)
    self.assertEqual(result.cpwer_insert, 1)
    self.assertEqual(result.cpwer_delete, 1)
    self.assertEqual(result.cpwer_sub, 0)
    self.assertEqual(result.cpwer_correct, 5)
    self.assertEqual(result.cpwer_total, 6)

  def test_wder_diff_words(self):
    hyp = "a b c d e f g h"
    ref = "a bb c e f gg g h ii"
    hyp_spk = "1 1 1 2 2 2 3 2"
    ref_spk = "2 2 2 2 3 3 4 3 2"
    result = metrics.compute_utterance_metrics(hyp, ref, hyp_spk, ref_spk)
    self.assertEqual(result.wer_correct, 6)
    self.assertEqual(result.wder_sub, 1)
    self.assertEqual(result.wder_correct, 6)
    self.assertEqual(result.wder_total, 7)
    self.assertEqual(result.cpwer_insert, 1)
    self.assertEqual(result.cpwer_delete, 2)
    self.assertEqual(result.cpwer_sub, 3)
    self.assertEqual(result.cpwer_correct, 4)
    self.assertEqual(result.cpwer_total, 9)

  def test_compute_metrics_on_json_dict(self):
    json_dict = {
        "utterances": [
            {
                "utterance_id": "utt1",
                "hyp_text": "hello good morning how are you",
                "hyp_spk": "1 1 1 2 2 2",
                "ref_text": "Hello. Good morning, how are you?",
                "ref_spk": "2 2 2 2 1 1",
            },
            {
                "utterance_id": "utt2",
                "hyp_text": "a b c d e f g h",
                "hyp_spk": "1 1 1 2 2 2 3 2",
                "ref_text": "a bb c e f gg g h ii",
                "ref_spk": "2 2 2 2 3 3 4 3 2",
            },
        ]
    }
    result = metrics.compute_metrics_on_json_dict(json_dict)
    self.assertEqual(result["utterances"][0]["utterance_id"], "utt1")
    self.assertEqual(result["utterances"][1]["utterance_id"], "utt2")
    self.assertAlmostEqual(result["WER"], 0.2666, delta=0.001)
    self.assertAlmostEqual(result["WDER"], 0.1538, delta=0.001)
    self.assertAlmostEqual(result["cpWER"], 0.5333, delta=0.001)

  def test_compute_metrics_on_json_dict_wer_only(self):
    json_dict = {
        "utterances": [
            {
                "utterance_id": "utt1",
                "hyp_text": "hello good morning how are you",
                "hyp_spk": "1 1 1 2 2 2",
                "ref_text": "Hello. Good morning, how are you?",
                "ref_spk": "2 2 2 2 1 1",
            },
            {
                "utterance_id": "utt2",
                "hyp_text": "a b c d e f g h",
                "hyp_spk": "1 1 1 2 2 2 3 2",
                "ref_text": "a bb c e f gg g h ii",
                "ref_spk": "2 2 2 2 3 3 4 3 2",
            },
        ]
    }
    result = metrics.compute_metrics_on_json_dict(
        json_dict, ref_spk_field="", hyp_spk_field=""
    )
    self.assertEqual(result["utterances"][0]["utterance_id"], "utt1")
    self.assertEqual(result["utterances"][1]["utterance_id"], "utt2")
    self.assertAlmostEqual(result["WER"], 0.2666, delta=0.001)

  def test_compute_metrics_on_json_file(self):
    json_file = os.path.join("testdata/example_data.json")
    with open(json_file, "r") as f:
        json_dict = json.load(f)
    result = metrics.compute_metrics_on_json_dict(json_dict)
    self.assertEqual(len(result["utterances"]), 2)
    self.assertEqual(result["utterances"][0]["utterance_id"], "en_0638")
    self.assertEqual(result["utterances"][1]["utterance_id"], "en_4157")
    self.assertAlmostEqual(result["WER"], 0.2363, delta=0.001)
    self.assertAlmostEqual(result["WDER"], 0.0437, delta=0.001)
    self.assertAlmostEqual(result["cpWER"], 0.2793, delta=0.001)

  def test_compute_metrics_on_json_file_oracle(self):
    json_file = os.path.join("testdata/example_data.json")
    with open(json_file, "r") as f:
        json_dict = json.load(f)
    result = metrics.compute_metrics_on_json_dict(
        json_dict, hyp_spk_field="hyp_spk_oracle"
    )
    self.assertEqual(len(result["utterances"]), 2)
    self.assertEqual(result["utterances"][0]["utterance_id"], "en_0638")
    self.assertEqual(result["utterances"][1]["utterance_id"], "en_4157")
    self.assertAlmostEqual(result["WER"], 0.2363, delta=0.001)
    self.assertAlmostEqual(result["WDER"], 0.0, delta=0.001)
    self.assertAlmostEqual(result["cpWER"], 0.2363, delta=0.001)

  def test_compute_metrics_on_json_file_degraded(self):
    json_file = os.path.join("testdata/example_data.json")
    with open(json_file, "r") as f:
        json_dict = json.load(f)
    result = metrics.compute_metrics_on_json_dict(
        json_dict, ref_spk_field="ref_spk_degraded"
    )
    self.assertEqual(len(result["utterances"]), 2)
    self.assertEqual(result["utterances"][0]["utterance_id"], "en_0638")
    self.assertEqual(result["utterances"][1]["utterance_id"], "en_4157")
    self.assertAlmostEqual(result["WER"], 0.2363, delta=0.001)
    self.assertAlmostEqual(result["WDER"], 0.0, delta=0.001)
    self.assertAlmostEqual(result["cpWER"], 0.2363, delta=0.001)


if __name__ == "__main__":
  unittest.main()
