import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.inference import TumorInferenceEngine


class TestInferenceImageLoading(unittest.TestCase):
    def test_load_rgb8_normalizes_channels(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)

            gray = np.random.randint(0, 256, size=(64, 64), dtype=np.uint8)
            rgb = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)

            gray_jpg = root / "gray.jpg"
            gray_png = root / "gray.png"
            rgb_jpg = root / "rgb.jpg"

            self.assertTrue(cv2.imwrite(str(gray_jpg), gray))
            self.assertTrue(cv2.imwrite(str(gray_png), gray))
            # cv2.imwrite expects BGR input for color files.
            self.assertTrue(cv2.imwrite(str(rgb_jpg), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)))

            for path in (gray_jpg, gray_png, rgb_jpg):
                loaded = TumorInferenceEngine.load_rgb8(str(path))
                self.assertEqual(loaded.dtype, np.uint8)
                self.assertEqual(loaded.shape, (64, 64, 3))

    def test_load_rgb8_unreadable_path_raises_clear_error(self):
        with self.assertRaisesRegex(ValueError, "unreadable or unsupported"):
            TumorInferenceEngine.load_rgb8("this_file_does_not_exist_12345.png")


if __name__ == "__main__":
    unittest.main()
