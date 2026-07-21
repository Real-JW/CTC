import unittest
import tempfile
from pathlib import Path

from BRIR.brir_fft_fir import (
    FftFirFilterBank,
    assess_brir_taps,
    compose_brir_filter_bank,
    compose_path_brir,
    design_demo_brir_filter_bank,
    design_demo_stereo_hrtf_filter_bank,
    load_reference_brir_filter_bank,
    fft_convolve,
    render_wav_with_brir,
    stereo_path_metadata,
    stereo_path_report,
)
from BRIR.wav import read_wav, write_wav_pcm16


class BrirFftFirTests(unittest.TestCase):
    def test_fft_convolve_matches_known_fir_result(self):
        self.assertEqual(fft_convolve([1.0, 2.0], [3.0, 4.0]), [3.0, 10.0, 8.0])

    def test_path_brir_composes_and_pads_to_requested_taps(self):
        brir = compose_path_brir([0.0, 1.0], [0.5, -0.25], taps=6)
        self.assertEqual(len(brir), 6)
        self.assertEqual(brir[:3], [0.0, 0.5, -0.25])

    def test_filter_bank_accepts_per_speaker_rtf(self):
        hrtf = [
            [[1.0, 0.0], [0.0, 0.5]],
            [[0.0, 0.5], [1.0, 0.0]],
        ]
        rtf = [[1.0, 0.25], [1.0, -0.25]]
        brir = compose_brir_filter_bank(hrtf, rtf, taps=5)
        self.assertEqual(len(brir), 2)
        self.assertEqual(len(brir[0]), 2)
        self.assertEqual(len(brir[0][0]), 5)
        self.assertAlmostEqual(brir[0][0][1], 0.25)
        self.assertAlmostEqual(brir[0][1][1], 0.5)

    def test_filter_bank_renderer_outputs_expected_tail(self):
        filters = [
            [[1.0, 0.5, 0.25], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, -0.5, 0.25]],
        ]
        renderer = FftFirFilterBank(filters, block_size=2)
        output = renderer.process_all([(1.0, 1.0)], include_tail=True)
        self.assertEqual(len(output), 3)
        self.assertAlmostEqual(output[0][0], 1.0)
        self.assertAlmostEqual(output[1][0], 0.5)
        self.assertAlmostEqual(output[2][0], 0.25)
        self.assertAlmostEqual(output[0][1], 1.0)
        self.assertAlmostEqual(output[1][1], -0.5)
        self.assertAlmostEqual(output[2][1], 0.25)

    def test_1024_tap_assessment_at_48k(self):
        assessment = assess_brir_taps(taps=1024, sample_rate=48_000, rt60_s=0.25)
        self.assertAlmostEqual(assessment.duration_ms, 21.333333, places=3)
        self.assertTrue(assessment.good_for_early_brir)
        self.assertFalse(assessment.captures_full_rt60)

    def test_demo_brir_shape_and_peak(self):
        brir = design_demo_brir_filter_bank(taps=1024, sample_rate=48_000)
        self.assertEqual(len(brir), 2)
        self.assertEqual(len(brir[0]), 2)
        self.assertEqual(len(brir[0][0]), 1024)
        peak = max(abs(tap) for row in brir for path in row for tap in path)
        self.assertLessEqual(peak, 0.99)

    def test_demo_hrtf_has_explicit_stereo_crosstalk_paths(self):
        hrtf = design_demo_stereo_hrtf_filter_bank(sample_rate=48_000)
        report = stereo_path_report(hrtf, sample_rate=48_000)
        roles = {entry["path"]: entry["role"] for entry in report}

        self.assertEqual(roles["left_speaker_to_left_ear"], "desired")
        self.assertEqual(roles["right_speaker_to_right_ear"], "desired")
        self.assertEqual(roles["right_speaker_to_left_ear"], "crosstalk")
        self.assertEqual(roles["left_speaker_to_right_ear"], "crosstalk")
        self.assertGreater(sum(abs(tap) for tap in hrtf[0][1]), 0.0)
        self.assertGreater(sum(abs(tap) for tap in hrtf[1][0]), 0.0)

    def test_stereo_path_metadata_names_all_four_paths(self):
        paths = stereo_path_metadata()
        self.assertEqual(len(paths), 4)
        self.assertEqual(sum(1 for path in paths if path["role"] == "desired"), 2)
        self.assertEqual(sum(1 for path in paths if path["role"] == "crosstalk"), 2)

    def test_reference_brir_uses_measured_kemar_coefficients(self):
        brir, metadata = load_reference_brir_filter_bank(taps=1024, sample_rate=48_000)
        report = stereo_path_report(brir, sample_rate=48_000)
        desired_peaks = [entry["peak"] for entry in report if entry["role"] == "desired"]
        crosstalk_peaks = [entry["peak"] for entry in report if entry["role"] == "crosstalk"]

        self.assertEqual(metadata["source"], "MIT Media Lab KEMAR HRTF compact measurements")
        self.assertEqual(len(brir[0][0]), 1024)
        self.assertGreater(min(desired_peaks), max(crosstalk_peaks))
        self.assertEqual(len(crosstalk_peaks), 2)

    def test_render_wav_with_brir_uses_input_output_paths(self):
        filters = [
            [[1.0, 0.25], [0.0, 0.0]],
            [[0.0, 0.0], [1.0, -0.25]],
        ]
        samples = [(0.5, -0.5), (0.25, -0.25)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.wav"
            output_path = root / "output_brir.wav"
            write_wav_pcm16(input_path, 48_000, samples)
            metrics = render_wav_with_brir(input_path, output_path, filters, sample_rate=48_000)
            sample_rate, rendered = read_wav(output_path)

        self.assertEqual(sample_rate, 48_000)
        self.assertEqual(metrics["input_samples"], 2)
        self.assertEqual(metrics["output_samples"], 3)
        self.assertEqual(len(metrics["paths"]), 4)
        self.assertEqual(len(rendered), 3)


if __name__ == "__main__":
    unittest.main()
