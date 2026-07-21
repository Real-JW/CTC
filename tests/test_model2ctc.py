import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from Model2CTC.config import DesignConfig
from Model2CTC.design import design_regularized_ctc, evaluate_ctc
from Model2CTC.export import export_fixed_header
from Model2CTC.io import load_filter, save_filter
from Model2CTC.model import ResidualModel, design_hybrid_ctc
from Model2CTC.training import train_residual_model


def synthetic_brir(cross_gain=0.35, cross_delay=5, taps=32):
    brir = np.zeros((2, 2, taps))
    brir[0, 0, 3] = 1.0
    brir[1, 1, 3] = 1.0
    brir[0, 1, cross_delay] = cross_gain
    brir[1, 0, cross_delay] = cross_gain
    return brir


def write_brir(path, brir, sample_rate=48_000):
    path.write_text(json.dumps({
        "sample_rate": sample_rate,
        "shape": list(brir.shape),
        "filters": brir.tolist(),
        "metadata": {},
    }))


class Model2CtcTests(unittest.TestCase):
    def setUp(self):
        self.config = DesignConfig(
            filter_taps=32,
            design_delay=12,
            fft_size=128,
            low_hz=200.0,
            high_hz=10_000.0,
            feature_bins=6,
            residual_components=3,
        )

    def test_regularized_design_is_bounded_and_reduces_crosstalk(self):
        brir = synthetic_brir()
        filters, metadata = design_regularized_ctc(brir, self.config)
        metrics = evaluate_ctc(brir, filters, self.config)

        self.assertEqual(filters.shape, (2, 2, 32))
        self.assertLessEqual(metrics["peak_filter_gain_db"], 6.01)
        self.assertGreater(metrics["median_crosstalk_suppression_db"], 15.0)
        self.assertIn("Tikhonov inverse", metadata["method"])

    def test_filter_json_and_fixed_point_export(self):
        filters, _ = design_regularized_ctc(synthetic_brir(), self.config)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            filter_path = root / "filter.json"
            header_path = root / "filter.h"
            save_filter(filter_path, 48_000, filters, {"test": True})
            rate, restored, metadata = load_filter(filter_path)
            export_fixed_header(restored, header_path)

            self.assertEqual(rate, 48_000)
            self.assertTrue(np.allclose(filters, restored))
            self.assertTrue(metadata["test"])
            self.assertIn("MODEL2CTC_TAPS 32", header_path.read_text())
            self.assertIn("MODEL2CTC_COEFF_Q 14", header_path.read_text())

    def test_train_and_apply_residual_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = []
            for index, gain in enumerate((0.25, 0.30, 0.35, 0.40)):
                predicted = synthetic_brir(cross_gain=gain * 0.8)
                measured = synthetic_brir(cross_gain=gain)
                predicted_path = root / f"predicted_{index}.json"
                measured_path = root / f"measured_{index}.json"
                write_brir(predicted_path, predicted)
                write_brir(measured_path, measured)
                cases.append({
                    "id": str(index),
                    "split": "test" if index == 3 else "train",
                    "predicted_brir": predicted_path.name,
                    "measured_brir": measured_path.name,
                })
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"cases": cases}))
            model_path = root / "model.json"

            report = train_residual_model(manifest, model_path, self.config)
            filters, metadata = design_hybrid_ctc(
                synthetic_brir(cross_gain=0.32),
                ResidualModel.load(model_path),
                self.config,
            )

            self.assertEqual(filters.shape, (2, 2, 32))
            self.assertEqual(report["training_cases"], 3)
            self.assertIn("metrics_on_predicted_brir", metadata)


if __name__ == "__main__":
    unittest.main()
