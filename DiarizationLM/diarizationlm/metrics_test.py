"""Tests for metrics."""

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


if __name__ == "__main__":
  unittest.main()
