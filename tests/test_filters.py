import unittest

from ctc.config import GeometryConfig, RoomConfig, RuntimeConfig
from ctc.filters import (
    FilterBankProcessor,
    build_direct_brir,
    design_analytic_filter_bank,
    direct_path_matrix,
    max_l1_gain,
)
from ctc.neural import ControllerContext, NeuralFilterController, apply_prediction


class FilterTests(unittest.TestCase):
    def test_direct_path_delays_match_default_geometry(self):
        paths = direct_path_matrix(GeometryConfig(), RuntimeConfig())
        self.assertAlmostEqual(paths[0][0].delay_samples, 70.0, delta=0.2)
        self.assertAlmostEqual(paths[0][1].delay_samples, 74.7, delta=0.2)

    def test_analytic_filter_bank_is_bounded(self):
        runtime = RuntimeConfig()
        filters = design_analytic_filter_bank(GeometryConfig(), runtime)
        self.assertEqual(len(filters), 2)
        self.assertEqual(len(filters[0]), 2)
        self.assertEqual(len(filters[0][0]), runtime.filter_taps)
        self.assertLessEqual(max_l1_gain(filters), 2.0)

    def test_ml_prediction_creates_valid_filter_bank(self):
        runtime = RuntimeConfig()
        base = design_analytic_filter_bank(GeometryConfig(), runtime)
        controller = NeuralFilterController()
        context = ControllerContext.from_block(
            [(0.1, -0.1)] * runtime.block_size,
            GeometryConfig(),
            RoomConfig(),
        )
        prediction = controller.predict(context)
        filters = apply_prediction(base, prediction, runtime, previous_filters=base)
        self.assertEqual(len(filters[0][0]), runtime.filter_taps)
        self.assertLessEqual(max_l1_gain(filters), 4.0)

    def test_static_brir_processor_outputs_tail(self):
        runtime = RuntimeConfig(brir_taps=96)
        brir = build_direct_brir(GeometryConfig(), runtime)
        processor = FilterBankProcessor(brir)
        output = processor.process_block([(1.0, 0.0)])
        output.extend(processor.flush(runtime.brir_taps - 1))
        self.assertEqual(len(output), runtime.brir_taps)
        self.assertGreater(max(abs(left) + abs(right) for left, right in output), 0.0)


if __name__ == "__main__":
    unittest.main()
