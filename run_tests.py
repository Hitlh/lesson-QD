"""Run repository tests without private checkpoints or datasets."""
from pathlib import Path
import unittest

if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    suite = unittest.defaultTestLoader.discover(str(root / 'backend/tests'))
    raise SystemExit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
