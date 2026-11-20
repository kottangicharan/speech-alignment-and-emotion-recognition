# Assignment 2b: Multimodal Emotion Recognition on TESS

This project recognizes **7 emotions** (angry, disgust, fear, happy, neutral, pleasant surprise, sad) in three ways:
1. **Speech only:** from the sound of the voice.
2. **Text only:** from the words that were said.
3. **Fusion:** from speech and text together.

It uses the **Toronto Emotional Speech Set (TESS)**: 2800 short recordings by two speakers, OAF (older, 64) and YAF (younger, 26). Each recording is the sentence "Say the word ___" spoken with one of the 7 emotions.

## How it works

Each system follows the block structure given in the assignment:

| Block | Speech | Text |
|---|---|---|
| **Preprocessing** | convert to 16 kHz mono, cut silence at the start and end, normalize the volume | build the sentence from the file name, lowercase, remove punctuation, split into tokens |
| **Feature extraction** | **WavLM-Large** (a large pretrained speech model, kept frozen) gives 1024 numbers for every 20 ms of audio | **RoBERTa-base** (a pretrained text model, kept frozen) gives 768 numbers per token |
| **Temporal / contextual modelling** | **BiLSTM** reads the frames forwards and backwards; **attention pooling** picks the most useful frames | **Transformer encoder** relates the tokens to each other; attention pooling |
| **Fusion** | **cross-modal attention** (speech looks at the text), then a **gate** that learns how much to trust speech vs text. A simpler version, plain concatenation, is also tested. | |
| **Classifier** | small neural network (MLP) → 7 emotions | |

The big pretrained models are **frozen** (not retrained). Their outputs are computed once and saved in `cache/`, so each experiment trains in minutes.

## How we test

- **Random split:** 70 % train, 15 % validation, 15 % test, repeated with 3 different random seeds. Both speakers appear in training and in test.
- **Speaker-independent split:** train on one speaker, test on the other one (OAF→YAF and YAF→OAF). This is harder and more honest: it checks whether the model learned *emotion*, or just *this person's voice*.

**Why text-only is at chance level (about 14 %):** every TESS sentence is "Say the word ___", and each of the 200 words is spoken in all 7 emotions. So the words never show the emotion. This is expected, and the report explains it.

## Setup

1. **Create the Python environment:**
   ```
   python -m venv .venv
   .venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
   pip install torch --index-url https://download.pytorch.org/whl/cu128
   pip install -r requirements.txt
   ```
   A GPU is recommended (we used an RTX 3050 with 6 GB). It also runs on CPU, more slowly.

2. **Download the data:** get TESS from Kaggle (`ejlok1/toronto-emotional-speech-set-tess`) and unzip it into `data/TESS/`. Or set the `TESS_DIR` environment variable to the unzipped folder. Any folder layout works: the code finds all `.wav` files, and reads the emotion from the file name (for example `OAF_back_angry.wav`; `ps` means pleasant surprise).

   **Data problems the code fixes automatically:**
   - The Kaggle zip contains the full dataset **twice** (5600 files). Duplicates are removed, leaving 2800.
   - `OA_bite_neutral.wav` has a typo in the speaker name. It is treated as OAF.
   - `YAF_germ_angry.wav` and `YAF_neat_fear.wav` are really AIFF files with a `.wav` name. They are read correctly anyway.
   - One file is recorded at 96 kHz and the rest at 24414 Hz. All are converted to 16 kHz.

## Run

Each `train.py` trains 5 models (3 random seeds + 2 speaker splits). Each `test.py` evaluates them and writes the results. Add `--smoke` to any `train.py` for a quick check that everything works (1 epoch on 64 clips).

1. **Speech only** (the first run also computes the WavLM features, about 10 minutes):
   ```
   python models/speech_pipeline/train.py
   python models/speech_pipeline/test.py
   ```
2. **Text only:**
   ```
   python models/text_pipeline/train.py
   python models/text_pipeline/test.py
   ```
3. **Fusion** (cross-modal attention + gate):
   ```
   python models/fusion_pipeline/train.py
   python models/fusion_pipeline/test.py
   ```
4. **Fusion with simple concatenation** (for comparison):
   ```
   python models/fusion_pipeline/train.py --fusion concat
   python models/fusion_pipeline/test.py --fusion concat
   ```
5. **Extra experiment, per-speaker normalization:** add `--spk_norm` to any speech or fusion command. Results are saved as `*_spknorm`.
6. **Report figure** (t-SNE grid): `python report_figures.py`

## Results

Test accuracy:

| Model | Random split (3 seeds) | Unseen speaker (mean of 2) |
|---|---|---|
| Speech | **100.00 ± 0.00 %** | **62.1 %** |
| Text | 14.37 ± 0.14 % (chance) | 14.3 % |
| Fusion (gated) | 99.92 ± 0.14 % | 59.6 % |
| Fusion (concat) | 99.44 ± 0.14 % | 56.4 % |

**Output files in `Results/`:**
- `accuracy_table.md` / `.csv`: accuracy and macro-F1 for every model and split, F1 for each emotion, and the fusion gate values.
- `<model>/<split>/confusion.png`: confusion matrix (which emotions get mixed up).
- `<model>/<split>/predictions.csv`, `metrics.json`: every prediction and the scores.
- `<model>/failure_cases.csv`: the most confident mistakes, used for the error analysis.
- `plots/tsne_grid.png` and `plots/tsne_*.png`, `plots/umap_*.png`: 2-D pictures of how well the emotions separate in the temporal, contextual and fusion blocks.
- `silhouette.csv`: one number per block for how well the emotion groups separate (1 = perfect).

The full analysis is in the combined report, [`REPORT.pdf`](../../REPORT.pdf) (Part 2).

## Project layout

```
models/common.py              shared code: data, features, models, training, testing, plots
models/speech_pipeline/       train.py, test.py
models/text_pipeline/         train.py, test.py
models/fusion_pipeline/       train.py, test.py
report_figures.py             makes the t-SNE figure for the report
Results/                      accuracy tables, confusion matrices, plots
requirements.txt              Python packages
```

**Not in the repository** (too large, rebuilt automatically): `data/` (download TESS), `cache/` (features, built on first run), `checkpoints/` (model weights, built by `train.py`).
