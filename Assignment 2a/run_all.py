"""Full MFA pipeline: prepare -> download models -> OOV + G2P custom dictionaries -> align 4 variants -> analyze.

Run inside the conda env that has MFA:   conda activate aligner && python run_all.py
Re-running keeps hand edits in dictionaries/oov_*.dict (delete a file to regenerate it with G2P).
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
os.environ.setdefault("MFA_ROOT_DIR", str(ROOT / "mfa_root"))  # project-local models, avoids OneDrive Documents
MFA_ROOT = Path(os.environ["MFA_ROOT_DIR"])
CORPUS, DICTS, OUT, WORK = ROOT / "corpus", ROOT / "dictionaries", ROOT / "outputs", ROOT / "work"
# wide beams: long 25-31 s BU utterances + non-native ISLE speech; 6 jobs = one per utterance
BEAM = ["--beam", "100", "--retry_beam", "400", "-j", str(min(6, os.cpu_count() or 1))]

MODELS = {
    "acoustic": ["english_us_arpa", "english_mfa"],
    "dictionary": ["english_us_arpa", "english_us_mfa", "english_nonnative_mfa"],
    "g2p": ["english_us_arpa", "english_us_mfa"],
}


def mfa(*args):
    cmd = ["mfa", *map(str, args)]
    print("\n>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def download_models():
    for kind, names in MODELS.items():
        for name in names:
            for attempt in range(3):  # downloads (~90 MB) can drop on flaky connections
                if list((MFA_ROOT / "pretrained_models" / kind).glob(name + ".*")):
                    break
                try:
                    mfa("model", "download", kind, name)
                except subprocess.CalledProcessError:
                    if attempt == 2:
                        raise
                    print(f"download of {kind} {name} failed, retrying...")


def build_dict(base, g2p_model, tag):
    """custom_<tag>.dict = pretrained dictionary + G2P pronunciations for this corpus's OOV words."""
    DICTS.mkdir(exist_ok=True)
    oov = DICTS / f"oov_{tag}.dict"
    if not oov.exists():
        found_dir = WORK / f"oovs_{tag}"
        shutil.rmtree(found_dir, ignore_errors=True)
        mfa("find_oovs", CORPUS, base, found_dir, "--clean")
        found = next(found_dir.rglob("oovs_found*.txt"), None)
        words = found.read_text(encoding="utf-8").split() if found else []
        print(f"OOVs for {base}: {words}")
        if words:
            mfa("g2p", found, g2p_model, oov, "--clean")
        else:
            oov.write_text("", encoding="utf-8")
    base_dict = next((MFA_ROOT / "pretrained_models" / "dictionary").glob(base + ".dict"))
    custom = DICTS / f"custom_{tag}.dict"
    custom.write_text(
        base_dict.read_text(encoding="utf-8").rstrip("\n") + "\n" + oov.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    mfa("validate", CORPUS, custom, "--ignore_acoustics", "--clean")  # should now report 0 OOVs
    return custom


def align(name, dictionary, acoustic):
    out = OUT / name
    shutil.rmtree(out, ignore_errors=True)
    mfa("align", CORPUS, dictionary, acoustic, out, "--clean", "--output_format", "long_textgrid", *BEAM)


def main():
    subprocess.run([sys.executable, str(ROOT / "prepare.py")], check=True)
    download_models()

    arpa = build_dict("english_us_arpa", "english_us_arpa", "arpa")
    mfa_us = build_dict("english_us_mfa", "english_us_mfa", "mfa")
    nonnative = build_dict("english_nonnative_mfa", "english_us_mfa", "nonnative")

    align("arpa", arpa, "english_us_arpa")
    align("mfa", mfa_us, "english_mfa")
    align("mfa_nonnative", nonnative, "english_mfa")

    # 4th variant: ARPA acoustic model adapted (MAP) to these 6 recordings, then aligned
    out = OUT / "arpa_adapted"
    shutil.rmtree(out, ignore_errors=True)
    WORK.mkdir(exist_ok=True)
    mfa("adapt", CORPUS, arpa, "english_us_arpa", WORK / "english_us_arpa_adapted.zip",
        "--output_directory", out, "--clean", *BEAM)

    subprocess.run([sys.executable, str(ROOT / "analyze.py")], check=True)


if __name__ == "__main__":
    main()
