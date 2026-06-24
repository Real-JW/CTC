import tempfile
import unittest
from pathlib import Path

from ctc.config import GeometryConfig, RoomConfig, RuntimeConfig
from ctc.neural import ControllerContext, load_controller
from ctc.training import train_linear_residual_model


class TrainingTests(unittest.TestCase):
    def test_training_writes_loadable_model(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "ml_filter_model.json"
            metrics = train_linear_residual_model(model_path, examples=24, ridge=1e-2)
            controller = load_controller(model_path)
            prediction = controller.predict(
                ControllerContext.from_block(
                    [(0.0, 0.0)] * RuntimeConfig().block_size,
                    GeometryConfig(),
                    RoomConfig(),
                )
            )
        self.assertEqual(metrics["output_count"], 72.0)
        self.assertEqual(len(prediction.residual_weights), 4)
        self.assertEqual(len(prediction.residual_weights[0]), 16)
        self.assertEqual(len(prediction.gains_db), 4)


if __name__ == "__main__":
    unittest.main()
