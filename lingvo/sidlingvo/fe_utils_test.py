from lingvo.core import py_utils
from lingvo.core import test_utils
import numpy as np

from sidlingvo import fe_utils


class FrontendTest(test_utils.TestCase):

  def test_frontend(self):
    """Test the frontend."""
    input_signal = np.random.randn(1, 16000)
    input_paddings = fe_utils.np.zeros(input_signal.shape[:2])

    fe = fe_utils.get_frontend_layer()
    result = fe.FProp(
        fe.theta,
        py_utils.NestedMap(src_inputs=input_signal, paddings=input_paddings),
    )
    features = result.src_inputs[0, :, :, 0].numpy()
    self.assertEqual(features.shape, (33, 512))
