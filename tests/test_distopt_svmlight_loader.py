import tempfile
import unittest
from pathlib import Path

import numpy as np


class TestSVMLightLoader(unittest.TestCase):
    def test_load_svmlight_tiny(self):
        try:
            from research.code.distopt.datasets.svmlight import load_svmlight
        except ImportError:
            self.skipTest("SciPy not installed")

        content = """1 1:2 3:4\n-1 2:5 3:6 # comment\n"""

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "tiny.svm"
            p.write_text(content)

            X, y = load_svmlight(p)

        self.assertEqual(X.shape[0], 2)
        self.assertEqual(y.shape, (2,))
        self.assertEqual(float(y[0]), 1.0)
        self.assertEqual(float(y[1]), -1.0)

        # n_features should be max index (=3) -> 3 columns
        self.assertEqual(X.shape[1], 3)

        Xd = X.toarray()
        self.assertTrue(np.allclose(Xd[0], np.array([2.0, 0.0, 4.0])))
        self.assertTrue(np.allclose(Xd[1], np.array([0.0, 5.0, 6.0])))


if __name__ == "__main__":
    unittest.main()
