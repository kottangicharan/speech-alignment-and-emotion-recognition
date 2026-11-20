# Assignment 2a: Forced Alignment with the Montreal Forced Aligner

**Forced alignment** means: we already know *what* was said (the transcript), and we want to find *when* each word and each sound was said in the audio. The result is a **TextGrid** file with a start and end time for every word and every phone (speech sound). It can be opened in Praat.

This folder aligns the **6 provided recordings** with the Montreal Forced Aligner (MFA):
- **3 news stories** from the BU Radio News Corpus (speaker `f2b`, 25–31 seconds each).
- **3 short sentences** from the ISLE corpus, spoken by a language learner ("I SAID WHITE NOT BAIT" and similar).

Only the provided data is used.

## 1. Install MFA

MFA uses the Kaldi speech toolkit, so it must be installed with **conda**. A plain `pip install` does not work.

1. Install **Miniforge** (a small version of conda): https://github.com/conda-forge/miniforge/releases. On Windows, pick `Miniforge3-Windows-x86_64.exe`.
2. Open **Miniforge Prompt** and run:
   ```
   conda create -n aligner -c conda-forge montreal-forced-aligner=3.4.4 -y
   conda activate aligner
   pip install praatio matplotlib numpy
   mfa version
   ```
   The last command should print the MFA version (3.4.x).
3. Optional: install **Praat** (https://www.fon.hum.uva.nl/praat/) to look at the results.

## 2. Prepare the data

```
python prepare.py
```

The provided data had three problems. `prepare.py` fixes them and writes a clean copy into `corpus/`. **The original files are never changed.**

| Problem | Fix |
|---|---|
| The 3 BU files end in `.wav` but are really **NIST SPHERE** audio, which MFA cannot read. | The file header is rewritten as a normal WAV header. The audio samples are copied byte for byte, and the script checks they are identical, so **no quality is lost**. |
| Some numbers are split by a hyphen and a space (`seventy- six`), which creates a fake word `seventy-`. | Changed to `seventy six` (3 places). **No other text is changed.** |
| MFA treats each folder as one speaker. | Files go into one folder per speaker: `corpus/f2b/` and `corpus/isle0131/`. |

MFA reads the uppercase `.TXT` transcripts without any change.

## 3. Run the alignment

**Everything in one command** (recommended):

```
conda activate aligner
python run_all.py
```

`run_all.py` runs these steps in order:

| Step | What happens | Example command |
|---|---|---|
| 1. Prepare data | runs `prepare.py` | `python prepare.py` |
| 2. Download models | gets the pretrained acoustic models, dictionaries and G2P models | `mfa model download acoustic english_us_arpa` |
| 3. Find missing words | lists words that are not in the dictionary (OOV, out-of-vocabulary) | `mfa find_oovs corpus english_us_arpa work/oovs_arpa` |
| 4. Guess pronunciations | a G2P model guesses how each missing word is pronounced from its spelling | `mfa g2p oovs.txt english_us_arpa dictionaries/oov_arpa.dict` |
| 5. Custom dictionary | base dictionary + our checked pronunciations, then a check for 0 missing words | `mfa validate corpus dictionaries/custom_arpa.dict --ignore_acoustics` |
| 6. Align | finds the word and phone times | `mfa align corpus dictionaries/custom_arpa.dict english_us_arpa outputs/arpa --beam 100 --retry_beam 400` |
| 7. Adapt and align | tunes the ARPA model to these speakers, then aligns again | `mfa adapt corpus dictionaries/custom_arpa.dict english_us_arpa work/adapted.zip --output_directory outputs/arpa_adapted` |
| 8. Analyze | compares the 4 alignments and makes tables and plots | `python analyze.py` |

Models are stored inside this folder (`mfa_root/`), not in your Documents folder.

**Checking pronunciations:** the guessed pronunciations are in `dictionaries/oov_*.dict`. We fixed one by hand: G2P read **WBUR** as one word ("why-bur"), but the radio station name is spelled letter by letter ("double-u B U R"). You can edit these files and run `python run_all.py` again; your edits are kept.

## 4. The four alignment setups

We tried several models and dictionaries to compare them (extra credit):

| Setup | Acoustic model | Dictionary |
|---|---|---|
| `arpa` | `english_us_arpa` | `english_us_arpa` + our missing words |
| `mfa` | `english_mfa` | `english_us_mfa` + our missing words |
| `mfa_nonnative` | `english_mfa` | `english_nonnative_mfa` + our missing words (made for learners' speech) |
| `arpa_adapted` | `english_us_arpa`, tuned on this data | our ARPA dictionary |

**Best setup: `arpa`.** It was the only one with no big errors. See the full report, [`REPORT.pdf`](../REPORT.pdf) (Part 1), for the full comparison.

## 5. Look at the results

**Output files:**
- `outputs/<setup>/<speaker>/*.TextGrid`: the main result, with a **words** tier and a **phones** tier.
- `results/summary.md`: statistics for each setup, how much the setups agree, and vowel lengths in the learner sentences.
- `results/anomalies.csv`: possible errors (phones only 10 ms long, very long words).
- `plots/praat_*.png`: Praat screenshots. `plots/*.png`: spectrograms with word and phone labels.

**Open a TextGrid in Praat:**
1. Praat → **Open → Read from file…** → choose `corpus/isle0131/ISLE_SESS0131_BLOCKD02_01_sprt1.wav`.
2. Again **Open → Read from file…** → choose `outputs/arpa/isle0131/ISLE_SESS0131_BLOCKD02_01_sprt1.TextGrid`.
3. Select both (Ctrl + click) → **View & Edit**. You will see the waveform, the spectrogram, and the word and phone boundaries.

## 6. Files

```
prepare.py       fixes the data (SPHERE to WAV, split numbers, speaker folders)
run_all.py       runs the full pipeline with one command
analyze.py       compares the alignments, makes tables and plots
wav/             provided audio (unchanged)
transcripts/     provided transcripts (unchanged)
dictionaries/    oov_*.dict = our checked pronunciations for missing words
outputs/         TextGrid files for each setup (main output)
results/         comparison tables
plots/           Praat screenshots and spectrogram plots
```
