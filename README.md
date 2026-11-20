# Speech Alignment and Emotion Recognition

**Assignment 2** for the Research Assistant position at the Speech Processing Lab, LTRC, IIIT Hyderabad.
**Author:** Charan Kottangi (kottangicharan117@gmail.com)

This repository has two parts:

| Part | What it does | Folder |
|---|---|---|
| **2a: Forced alignment** | Finds the exact start and end time of every word and every sound in 6 recordings, using the Montreal Forced Aligner (MFA). | [`Assignment 2a/`](Assignment%202a/) |
| **2b: Emotion recognition** | Recognizes 7 emotions from speech, from text, and from both together, using the TESS dataset. | [`Assignment 2b/project/`](Assignment%202b/project/) |

**Full report (both parts):** [`REPORT.pdf`](REPORT.pdf)

---

## Main results

**2a: Forced alignment**
- All 6 recordings were aligned, and every word got a time. Output: 24 TextGrid files (6 recordings × 4 model setups).
- On clear news speech, two different models agree to within 7–10 ms.
- On the noisy learner recordings, one model (`english_mfa`) wrongly stretched the last word over about 2 seconds of background noise. The other model (`english_us_arpa`) got it right.
- **Best setup:** `english_us_arpa` model with our custom dictionary.

**2b: Emotion recognition**

| Model | Random split | New, unseen speaker |
|---|---|---|
| Speech only | **100.0 %** | **62.1 %** |
| Text only | 14.4 % (chance level) | 14.3 % |
| Speech + text (fusion) | 99.9 % | 59.6 % |

- Speech works very well when the speaker was also in training.
- Text alone cannot work on TESS: every sentence is "Say the word ___", so the words never show the emotion.
- The hard problem is a new speaker: accuracy drops to about 60 %, because each speaker shows emotions with a different pitch.

---

## Quick start

### 2a: Forced alignment

You need MFA, which installs through conda (not plain pip).

```
conda create -n aligner -c conda-forge montreal-forced-aligner=3.4.4 -y
conda activate aligner
pip install praatio matplotlib numpy
cd "Assignment 2a"
python run_all.py
```

One command runs everything: data preparation → model download → custom dictionary → 4 alignments → analysis. The TextGrid files appear in `Assignment 2a/outputs/`. Step-by-step details are in [`Assignment 2a/README.md`](Assignment%202a/README.md).

### 2b: Emotion recognition

```
cd "Assignment 2b/project"
python -m venv .venv
.venv\Scripts\activate                     # Linux/macOS: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Download TESS from Kaggle (`ejlok1/toronto-emotional-speech-set-tess`) and unzip it into `data/TESS/`. Then train and test each model:

```
python models/speech_pipeline/train.py
python models/speech_pipeline/test.py
python models/text_pipeline/train.py
python models/text_pipeline/test.py
python models/fusion_pipeline/train.py
python models/fusion_pipeline/test.py
```

Results (tables and plots) appear in `Assignment 2b/project/Results/`. Details are in [`Assignment 2b/project/README.md`](Assignment%202b/project/README.md).

---

## Repository layout

```
REPORT.pdf                      full report for both parts
Assignment 2a/
  README.md                     how to install MFA, prepare data and run alignment
  prepare.py                    fixes the data so MFA can read it
  run_all.py                    runs the whole pipeline with one command
  analyze.py                    compares the alignments, makes tables and plots
  wav/, transcripts/            the provided recordings and their text
  dictionaries/oov_*.dict       our hand-checked pronunciations for missing words
  outputs/<setup>/              the TextGrid files (main output)
  results/                      comparison tables
  plots/                        Praat screenshots and spectrogram plots
Assignment 2b/project/
  README.md                     how to set up, train and test
  models/common.py              shared code: data, features, models, training
  models/speech_pipeline/       train.py, test.py
  models/text_pipeline/         train.py, test.py
  models/fusion_pipeline/       train.py, test.py
  Results/                      accuracy tables, confusion matrices, plots
  requirements.txt              Python packages
```

## Not included in this repository (too large, and rebuilt automatically)

| Item | Size | How to get it back |
|---|---|---|
| TESS dataset | about 400 MB | Download from Kaggle (see above) |
| MFA models and temporary files (`mfa_root/`, `work/`) | 1.4 GB | `run_all.py` downloads and rebuilds them |
| Full merged dictionaries (`custom_*.dict`) | 12 MB | `run_all.py` rebuilds them from the base dictionary + `oov_*.dict` |
| Cached speech and text features (`cache/`) | 3.2 GB | Built on the first run of any `train.py` |
| Trained model weights (`checkpoints/`) | 440 MB | Built by each `train.py` |
