"""Build an MFA-ready corpus in ./corpus from ./wav and ./transcripts.

- NIST SPHERE .wav (BU Radio Corpus) -> real RIFF WAV, samples copied byte-for-byte.
- Already-RIFF .wav (ISLE) copied unchanged.
- Transcripts: only fix split numbers ("seventy- six" -> "seventy six"). File names untouched.
Originals are never modified.
"""
import re
import shutil
import wave
from pathlib import Path

ROOT = Path(__file__).parent
SRC_WAV, SRC_TXT, OUT = ROOT / "wav", ROOT / "transcripts", ROOT / "corpus"


def read_nist(path):
    data = path.read_bytes()
    assert data[:7] == b"NIST_1A", path
    header_size = int(data[8:16])
    fields = {}
    for line in data[16:header_size].decode("ascii", "ignore").splitlines():
        parts = line.split()
        if len(parts) == 3:
            fields[parts[0]] = parts[2]
    assert fields["sample_coding"] == "pcm", "compressed SPHERE not supported"
    assert fields["sample_byte_format"] == "01", "expected little-endian"  # same as WAV
    n_bytes, rate, ch = int(fields["sample_n_bytes"]), int(fields["sample_rate"]), int(fields["channel_count"])
    pcm = data[header_size:]
    assert len(pcm) == int(fields["sample_count"]) * n_bytes * ch, "size mismatch"
    return pcm, rate, ch, n_bytes


def nist_to_wav(src, dst):
    pcm, rate, ch, n_bytes = read_nist(src)
    with wave.open(str(dst), "wb") as w:
        w.setnchannels(ch)
        w.setsampwidth(n_bytes)
        w.setframerate(rate)
        w.writeframes(pcm)
    # lossless check: samples read back must be identical to the original bytes
    with wave.open(str(dst), "rb") as w:
        assert w.readframes(w.getnframes()) == pcm, f"{dst} not bit-identical"
    return len(pcm) // (n_bytes * ch) / rate


def fix_split_numbers(text):
    # "seventy- six" -> "seventy six"; leaves "de-politicize" alone (no space after hyphen)
    return re.sub(r"(\w)-\s+(\w)", r"\1 \2", text)


def speaker_dir(name):
    # MFA treats each subfolder as one speaker
    d = OUT / ("f2b" if name.upper().startswith("F2B") else "isle0131")
    d.mkdir(parents=True, exist_ok=True)
    return d


def main():
    shutil.rmtree(OUT, ignore_errors=True)  # corpus/ is generated output only
    for src in sorted(SRC_WAV.glob("*.wav")):
        dst = speaker_dir(src.name) / src.name
        if src.read_bytes()[:4] == b"RIFF":
            shutil.copy2(src, dst)
            print(f"copied     {src.name}")
        else:
            secs = nist_to_wav(src, dst)
            print(f"converted  {src.name}  ({secs:.2f}s, bit-identical)")
    for src in sorted(SRC_TXT.iterdir()):
        text = src.read_text(encoding="utf-8", errors="replace")
        fixed = fix_split_numbers(text)
        (speaker_dir(src.name) / src.name).write_text(fixed, encoding="utf-8")
        n =len(re.findall(r"\w-\s+\w", text))
        print(f"transcript {src.name}" + (f"  ({n} split number(s) fixed)" if n else ""))


if __name__ == "__main__":
    assert fix_split_numbers("seventy- six and de-politicize") == "seventy six and de-politicize"
    main()
