"""One command: train all models, then regenerate the write-up PDF and slide deck.
   python run_all.py --data_dir data --team "Alice (PES1..), Bob (PES1..) | Sec A"  [--fast]"""
import argparse
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", default="data"); ap.add_argument("--results", default="results")
ap.add_argument("--team", default="[Member 1 - SRN] | [Member 2 - SRN] | Section [X]")
ap.add_argument("--fast", action="store_true")
a = ap.parse_args()
cmd = [sys.executable, "-m", "src.train", "--data_dir", a.data_dir, "--out", a.results] + (["--fast"] if a.fast else [])
subprocess.check_call(cmd)
subprocess.check_call([sys.executable, "docs/build_writeup.py", "--results", a.results, "--out", "writeup.pdf", "--team", a.team])
subprocess.check_call([sys.executable, "docs/build_slides.py", "--results", a.results, "--out", "slides.pptx", "--team", a.team])
print("Done: writeup.pdf, slides.pptx, results/submission.csv")
