"""Text pipeline: train. Usage: python models/text_pipeline/train.py [--splits all|random|speaker] [--smoke]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import train_main

if __name__ == "__main__":
    train_main("text")
