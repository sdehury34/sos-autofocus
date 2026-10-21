"""
Download the Waterloo calf subset of OpenH-RF (CC-BY-4.0) that has a through-transmission speed of sound:
one file for each participant and leg (lateral, longitudinal, relaxed, minimal pressure).
31 participants, 62 files, 28.6 GB (26.6 GiB). The download resumes: run it again after a stop.

The dataset is gated on Hugging Face: accept its conditions once on the dataset page while logged in,
then give a read token in the environment variable HF_TOKEN.

    HF_TOKEN=hf_... python download_waterloo_calf.py --out data/waterloo_calf
"""
import argparse, json, os, re, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

API = "https://huggingface.co/api/datasets/nvidia/OpenH-RF/tree/main/waterloo-muscle/data?recursive=true"
BASE = "https://huggingface.co/datasets/nvidia/OpenH-RF/resolve/main/"
AUTH = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if os.environ.get("HF_TOKEN") else {}
RE = re.compile(r"Acq_(p\d+)_Calf_(left|right)_calf_(\w+?)_(longitudinal|transverse)_(\w+?)_(minimal|pressure)\.hdf5")


def selection():
    files = [x for x in json.load(urllib.request.urlopen(urllib.request.Request(API, headers=AUTH), timeout=60)) if x["type"] == "file" and "Calf" in x["path"]]
    groups = {}
    for x in files:
        m = RE.search(x["path"])
        if m: groups.setdefault((m[1], m[2]), []).append((x, m))
    score = lambda m: (m[3] == "lateral") * 8 + (m[4] == "longitudinal") * 4 + (m[5] == "relaxed") * 2 + (m[6] == "minimal")
    return [max(g, key=lambda t: score(t[1]))[0] for _, g in sorted(groups.items())]


def fetch(path, size, out):
    dst = out / Path(path).name
    for attempt in range(50):
        have = dst.stat().st_size if dst.exists() else 0
        if have >= size: return dst
        try:
            req = urllib.request.Request(BASE + path, headers={"Range": f"bytes={have}-", **AUTH})
            with urllib.request.urlopen(req, timeout=60) as r, open(dst, "ab") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b: break
                    f.write(b)
        except Exception as e:
            print(f"  retry {attempt} ({type(e).__name__}) at {dst.stat().st_size / 1e6:.0f} MB", flush=True); time.sleep(5)
    raise RuntimeError(f"download failed: {path}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, default=Path("data/waterloo_calf"))
    ap.add_argument("--workers", type=int, default=4, help="parallel downloads")
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    sel = selection(); total = sum(x["lfs"]["size"] if "lfs" in x else x["size"] for x in sel)
    print(f"{len(sel)} files, {total / 1e9:.1f} GB -> {a.out}", flush=True)
    t = time.time(); size = lambda x: x["lfs"]["size"] if "lfs" in x else x["size"]
    with ThreadPoolExecutor(a.workers) as ex:
        jobs = {ex.submit(fetch, x["path"], size(x), a.out): x for x in sel}
        for i, j in enumerate(as_completed(jobs), 1):
            x = jobs[j]; j.result()
            print(f"[{i}/{len(sel)}] {Path(x['path']).name} ({size(x) / 1e6:.0f} MB, {time.time() - t:.0f} s)", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
