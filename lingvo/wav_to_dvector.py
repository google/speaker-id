"""A library to run speaker-id TFLite model inference."""

import dataclasses
from lingvo.core import py_utils
import numpy as np
import fe_utils


@dataclasses.dataclass
class WavToDvectorRunner:
  """WavToDvectorRunner."""

  # Path to the VAD TFLite model.
  vad_model_file: str = ""
  # Path to the VAD mean and stddev CSV file.
  vad_mean_stddev_file: str = ""
  # Path to the speaker-id TFLite model.
  tisid_model_file: str = ""

  vad_threshold: float = 0.1
  vad_cluster_id: int = 2
  vad_num_clusters: int = 16

  def __post_init__(self):
    self.vad_model = fe_utils.load_tflite_model(self.vad_model_file)
    self.tisid_model = fe_utils.load_tflite_model(self.tisid_model_file)
    self.vad_mean_stddev = fe_utils.read_mean_stddev_csv(
        self.vad_mean_stddev_file
    )

  def wav_to_dvector(self, audio_file: str) -> np.ndarray:
    """Run speaker-id model on wav file."""
    input_signal = fe_utils.get_int_samples(audio_file)
    return self.samples_to_dvector(input_signal)

  def samples_to_dvector(self, input_signal: np.ndarray) -> np.ndarray:
    """Run speaker-id model on int16 samples."""
    input_paddings = fe_utils.np.zeros(input_signal.shape[:2])
    fe = fe_utils.get_frontend_layer()
    result = fe.FProp(
        fe.theta,
        py_utils.NestedMap(src_inputs=input_signal, paddings=input_paddings),
    )
    features = result.src_inputs[0, :, :, 0].numpy()
    self.vad_model.reset_all_variables()
    _, features_after_vad = fe_utils.apply_vad(
        features,
        self.vad_model,
        self.vad_mean_stddev,
        self.vad_threshold,
        self.vad_cluster_id,
        self.vad_num_clusters,
    )
    self.tisid_model.reset_all_variables()
    dvectors = fe_utils.run_multi_input_model(
        self.tisid_model, features_after_vad
    )
    return dvectors
