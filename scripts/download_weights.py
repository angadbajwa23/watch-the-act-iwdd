"""Download the trained weights of the IWDD 2027 submission (E0 members + heads, ~300 MB) and check their SHA-256.

  python scripts/download_weights.py            # -> weights/
The two frozen encoders (SigLIP2, V-JEPA 2) are not in this archive; iwdd.system downloads them from Hugging Face
at pinned revisions on first use.
"""
import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://github.com/angadbajwa23/watch-the-act-iwdd/releases/download/v1.0/iwdd2027_weights.zip"
SHA256 = "ab52bd47d65c38432968f02a17809d4c782fafa5b86d5c202df3e3429611c919"
REPO = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=URL)
    ap.add_argument("--sha256", default=SHA256, help="expected checksum of the archive")
    ap.add_argument("--out", default=str(REPO / "weights"))
    a = ap.parse_args()
    out = Path(a.out)
    zpath = out.with_suffix(".zip")
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {a.url}", flush=True)
    urllib.request.urlretrieve(a.url, zpath)
    got = sha256(zpath)
    if got != a.sha256:
        zpath.unlink()
        sys.exit(f"SHA-256 mismatch: got {got}, expected {a.sha256}")
    with zipfile.ZipFile(zpath) as z:
        z.extractall(out)
    zpath.unlink()
    print(f"weights in {out}: " + ", ".join(sorted(p.name for p in out.iterdir())))


if __name__ == "__main__":
    main()
