import tempfile
import unittest
from pathlib import Path

from ctc.evaluate import evaluate_ctc_model, evaluate_pipeline
from ctc.config import GeometryConfig, RoomConfig, RuntimeConfig
from ctc.pipeline import render_stage1, render_stage2
from ctc.training import train_linear_residual_model
from ctc.wav import write_wav_pcm16


class EvaluateTests(unittest.TestCase):
    def test_ctc_model_metrics_include_suppression_and_latency(self):
        metrics = evaluate_ctc_model(
            input_samples=[(0.1, -0.1)] * 128,
            geometry=GeometryConfig(),
            room=RoomConfig(),
            runtime=RuntimeConfig(),
            model_path=None,
        )
        self.assertIn("overall_worst_suppression_db", metrics)
        self.assertIn("band_suppression", metrics)
        self.assertIn("desired_latency_mean_ms", metrics)
        self.assertGreater(float(metrics["desired_latency_mean_ms"]), 0.0)
        self.assertEqual(len(metrics["band_suppression"]), 7)

    def test_evaluate_pipeline_writes_metrics_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.wav"
            stage1_path = root / "preprocessed.wav"
            stage2_path = root / "simulated_binaural.wav"
            model_path = root / "ml_filter_model.json"
            metrics_path = root / "metrics.json"

            write_wav_pcm16(input_path, 48_000, [(0.2, -0.2)] * 256)
            train_linear_residual_model(model_path, examples=24, ridge=1e-2)
            render_stage1(input_path, stage1_path, model_path=model_path)
            render_stage2(stage1_path, stage2_path)
            report = evaluate_pipeline(
                input_path=input_path,
                stage1_path=stage1_path,
                stage2_path=stage2_path,
                model_path=model_path,
                output_path=metrics_path,
            )
            metrics_exists = metrics_path.exists()

        self.assertTrue(metrics_exists)
        self.assertIn("files", report)
        self.assertIn("ctc_model", report)
        self.assertIn("acceptance", report)


if __name__ == "__main__":
    unittest.main()
