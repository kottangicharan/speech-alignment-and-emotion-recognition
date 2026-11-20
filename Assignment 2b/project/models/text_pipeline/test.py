"""Text pipeline: test. Usage: python models/text_pipeline/test.py [--splits all|random|speaker]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import test_main

if __name__ == "__main__":
    test_main("text")
