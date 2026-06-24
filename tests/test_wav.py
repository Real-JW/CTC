import tempfile
import unittest
from pathlib import Path

from ctc.wav import read_wav, write_wav_float32, write_wav_pcm16


class WavTests(unittest.TestCase):
    def test_float32_round_trip(self):
        samples = [(0.0, 0.25), (-0.5, 0.5), (1.25, -1.25)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.wav"
            write_wav_float32(path, 48_000, samples)
            sample_rate, decoded = read_wav(path)
        self.assertEqual(sample_rate, 48_000)
        self.assertEqual(len(decoded), len(samples))
        self.assertAlmostEqual(decoded[1][0], -0.5, places=6)
        self.assertAlmostEqual(decoded[2][1], -1.25, places=6)

    def test_pcm16_round_trip_is_stereo(self):
        samples = [(0.1, -0.1), (0.5, -0.5)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.wav"
            write_wav_pcm16(path, 48_000, samples)
            sample_rate, decoded = read_wav(path)
        self.assertEqual(sample_rate, 48_000)
        self.assertEqual(len(decoded), 2)
        self.assertAlmostEqual(decoded[0][0], 0.1, places=3)
        self.assertAlmostEqual(decoded[0][1], -0.1, places=3)


if __name__ == "__main__":
    unittest.main()
