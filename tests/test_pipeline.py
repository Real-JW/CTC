import tempfile
import unittest
from pathlib import Path

from ctc.pipeline import render_stage1, render_stage2, run_pipeline
from ctc.training import train_linear_residual_model
from ctc.wav import read_wav, write_wav_pcm16


class PipelineTests(unittest.TestCase):
    def test_stage1_and_stage2_render_files(self):
        samples = [(0.25, -0.25)] * 300
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.wav"
            stage1_path = root / "preprocessed.wav"
            stage2_path = root / "simulated_binaural.wav"
            model_path = root / "ml_filter_model.json"
            write_wav_pcm16(input_path, 48_000, samples)
            train_linear_residual_model(model_path, examples=24, ridge=1e-2)

            stage1_metrics = render_stage1(input_path, stage1_path, model_path=model_path)
            stage2_metrics = render_stage2(stage1_path, stage2_path)
            stage1_rate, stage1 = read_wav(stage1_path)
            stage2_rate, stage2 = read_wav(stage2_path)

        self.assertEqual(stage1_rate, 48_000)
        self.assertEqual(stage2_rate, 48_000)
        self.assertEqual(len(stage1), len(samples))
        self.assertGreater(len(stage2), len(stage1))
        self.assertEqual(stage1_metrics["ml_model_loaded"], 1.0)
        self.assertGreaterEqual(stage2_metrics["output_samples"], stage1_metrics["output_samples"])

    def test_run_pipeline_default_controller(self):
        samples = [(0.1, 0.1)] * 130
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.wav"
            stage1_path = root / "preprocessed.wav"
            stage2_path = root / "simulated_binaural.wav"
            write_wav_pcm16(input_path, 48_000, samples)
            metrics = run_pipeline(input_path, stage1_path, stage2_path)
        self.assertEqual(metrics["stage1"]["output_samples"], float(len(samples)))
        self.assertGreater(metrics["stage2"]["output_samples"], metrics["stage1"]["output_samples"])


if __name__ == "__main__":
    unittest.main()
