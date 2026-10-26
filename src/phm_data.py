"""Download, extract, and flatten the PHM 2026 Data Challenge datasets from the GTC Synology shares.

Layout produced (one flat directory per split x modality, no nested folders or zips):

    data/<split>/<modality>/<Exp-X>_<Run>_<original_name>
    data/<split>/manifest.csv      (one row per extracted file, with provenance)
    data/<split>/*.csv             (submission templates, copied as-is)

modality: lf (CIs + oil + environment), ci (CIs only), hf (high-frequency), photos

Usage:
    python scripts/phm_data.py small                    # LF, CI, photos, CSVs (~1.7 GB)
    python scripts/phm_data.py hf [--split train] [--exp Exp-A] [--run Run-1] [--dest DIR] [--workers 8]
        HF members are streamed straight out of the remote zip via HTTP range requests,
        so the ~20 GB zips never touch disk. Stops when free space runs low.
"""

import argparse
import csv
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.parse
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HOST = "https://gtc-data.synology.me:51111"
SHARES = {
    "train": {"id": "uIrAvzqEh", "root": "/train"},
    "test": {
        "id": "3LlAuvblO",
        "root": "/test",
        "password_env": "PHM_TEST_SHARE_PASSWORD",
    },
    "validation": {"id": "FpZSWQYD3", "root": "/validation"},
}
ROOT = Path(__file__).resolve().parent.parent / "data"
RAW = ROOT / "raw"
JUNK = {"Thumbs.db", ".DS_Store"}
MIN_FREE = 5 * 1024**3  # keep 5 GB headroom


# ---------------------------------------------------------------- Synology API


class Share:
    def __init__(self, split):
        cfg = SHARES[split]
        self.split, self.id, self.root = split, cfg["id"], cfg["root"]
        self.jar = Path(tempfile.gettempdir()) / f"phm_{self.id}.cookies"
        self._curl("-c", str(self.jar), f"{HOST}/sharing/{self.id}", "-o", os.devnull)
        if "password_env" in cfg:
            password = os.environ.get(cfg["password_env"])
            if not password:
                raise SystemExit(
                    f"the {split} share needs a password: export {cfg['password_env']}=<password from the challenge>"
                )
            out = self._post(
                f"{HOST}/sharing/webapi/entry.cgi",
                api="SYNO.Core.Sharing.Login",
                version=1,
                method="login",
                sharing_id=json.dumps(self.id),
                password=json.dumps(password),
            )
            sid = out["data"]["sharing_sid"]
            self.jar.write_text(
                f"{urllib.parse.urlparse(HOST).hostname}\tFALSE\t/\tFALSE\t0\tsharing_sid\t{sid}\n"
            )

    def _curl(self, *args, capture=True):
        return subprocess.run(
            ["curl", "-sk", "-b", str(self.jar), *args],
            capture_output=capture,
            check=True,
        ).stdout

    def _post(self, url, **params):
        args = []
        for k, v in params.items():
            args += ["--data-urlencode", f"{k}={v}"]
        out = json.loads(self._curl(url, *args))
        if not out.get("success"):
            raise RuntimeError(f"{params.get('api')} failed: {out}")
        return out

    def ls(self, folder):
        files, offset = [], 0
        while True:
            d = self._post(
                f"{HOST}/webapi/entry.cgi",
                api="SYNO.FolderSharing.List",
                version=2,
                method="list",
                offset=offset,
                limit=5000,
                sort_by="name",
                action="enum",
                filetype="all",
                _sharing_id=json.dumps(self.id),
                folder_path=json.dumps(folder),
                additional='["size"]',
            )["data"]
            files += d["files"]
            offset += len(d["files"])
            if offset >= d["total"] or not d["files"]:
                return files

    def walk(self, folder=None):
        for f in self.ls(folder or self.root):
            if f["isdir"]:
                yield from self.walk(f["path"])
            else:
                yield f["path"], f["additional"]["size"]

    def url(self, path):
        name = path.rsplit("/", 1)[1]
        q = urllib.parse.urlencode(
            {
                "api": "SYNO.FolderSharing.Download",
                "version": 2,
                "method": "download",
                "mode": "download",
                "stdhtml": "false",
                "dlname": json.dumps(name),
                "path": json.dumps([path]),
                "_sharing_id": json.dumps(self.id),
            }
        )
        return (
            f"{HOST}/fsdownload/webapi/file_download.cgi/{urllib.parse.quote(name)}?{q}"
        )

    def download(self, path, size, dest):
        if dest.exists() and dest.stat().st_size == size:
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(5):
            subprocess.run(
                [
                    "curl",
                    "-sk",
                    "-L",
                    "--retry",
                    "5",
                    "-C",
                    "-",
                    "-b",
                    str(self.jar),
                    "-o",
                    str(dest),
                    self.url(path),
                ]
            )
            if dest.exists() and dest.stat().st_size == size:
                return
        raise RuntimeError(f"download failed: {path}")


class RemoteFile(io.RawIOBase):
    """Seekable read-only view of a remote file using HTTP range requests (lets zipfile stream members)."""

    def __init__(self, share, path, size):
        self.share, self.url, self.size, self.pos = share, share.url(path), size, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = (
            offset
            if whence == 0
            else self.pos + offset
            if whence == 1
            else self.size + offset
        )
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        for _ in range(5):
            data = self.share._curl(
                "--retry", "5", "-r", f"{self.pos}-{self.pos + n - 1}", self.url
            )
            if len(data) == n:
                break
        else:
            raise IOError(f"short range read at {self.pos}")
        b[:n] = data
        self.pos += n
        return n


# ---------------------------------------------------------------- naming


def norm_run(label):
    """'Run 1' / 'Run-1' -> 'Run-1'; 'Break in' -> 'Break-In'; '0 Hours - Test Start' -> 'Test-Start'."""
    s = label.strip()
    if re.search(r"test start", s, re.I):
        return "Test-Start"
    if re.fullmatch(r"break[\s-]*in", s, re.I):
        return "Break-In"
    if re.fullmatch(r"pre[\s-]*run", s, re.I):
        return "Pre-Run"
    m = re.fullmatch(r"run[\s-]*(\d+)", s, re.I)
    if m:
        return f"Run-{int(m.group(1))}"
    return re.sub(r"\s+", "-", s)


def clean(name):
    return re.sub(r"\s+", "_", name.strip())


def exp_of(name):
    return re.match(r"(Exp-[A-Z])", name, re.I).group(1).title().replace("exp", "Exp")


class Writer:
    def __init__(self, split, dest_root):
        self.split, self.dir = split, dest_root / split
        self.dir.mkdir(parents=True, exist_ok=True)
        self.manifest = self.dir / "manifest.csv"
        self.rows = {}
        if self.manifest.exists():
            with open(self.manifest) as f:
                self.rows = {r["file"]: r for r in csv.DictReader(f)}

    def put(self, modality, exp, run, inner_name, src, source_zip, inner_path, size):
        """src: bytes, or file-like to stream. Returns destination path."""
        out = self.dir / modality / f"{exp}_{run}_{clean(inner_name)}"
        rel = str(out.relative_to(self.dir))
        if rel in self.rows and out.exists() and out.stat().st_size == int(size):
            return out
        if rel in self.rows and self.rows[rel]["inner_path"] != inner_path:
            raise RuntimeError(
                f"name collision: {rel} from {inner_path} and {self.rows[rel]['inner_path']}"
            )
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".part")
        with open(tmp, "wb") as f:
            if isinstance(src, bytes):
                f.write(src)
            else:
                shutil.copyfileobj(src, f, 16 * 1024**2)
        tmp.rename(out)
        self.rows[rel] = dict(
            file=rel,
            modality=modality,
            experiment=exp,
            run=run,
            size=size,
            source_zip=source_zip,
            inner_path=inner_path,
        )
        return out

    def save(self):
        fields = [
            "file",
            "modality",
            "experiment",
            "run",
            "size",
            "source_zip",
            "inner_path",
        ]
        with open(self.manifest, "w", newline="") as f:
            w = csv.DictWriter(f, fields)
            w.writeheader()
            w.writerows(sorted(list(self.rows.values()), key=lambda r: r["file"]))


def members(zf):
    for info in zf.infolist():
        base = info.filename.rstrip("/").rsplit("/", 1)[-1]
        if (
            info.is_dir()
            or base in JUNK
            or "__MACOSX" in info.filename
            or base.startswith("._")
        ):
            continue
        yield info, base


# ---------------------------------------------------------------- small files


def flatten_small(split):
    share, w = Share(split), Writer(split, ROOT)
    for path, size in share.walk():
        if "/high-frequency/" in path:
            continue
        local = RAW / path.lstrip("/")
        share.download(path, size, local)
        rel = path[len(share.root) + 1 :]
        if path.endswith(".csv"):
            shutil.copy2(local, w.dir / local.name)
            continue
        if "/low-frequency" in path:
            modality = "ci" if "(CIs)" in path else "lf"
            with zipfile.ZipFile(local) as outer:
                for inner, inner_base in members(outer):
                    if not inner_base.endswith(".zip"):
                        raise RuntimeError(
                            f"unexpected non-zip in {rel}: {inner.filename}"
                        )
                    exp = exp_of(inner_base)
                    run = norm_run(re.search(r"(Run-\d+)", inner_base).group(1))
                    with zipfile.ZipFile(io.BytesIO(outer.read(inner))) as z:
                        for info, base in members(z):
                            w.put(
                                modality,
                                exp,
                                run,
                                base,
                                z.read(info),
                                rel,
                                f"{inner.filename}/{info.filename}",
                                info.file_size,
                            )
        elif "/photos/" in path:
            exp = exp_of(local.name)
            with zipfile.ZipFile(local) as z:
                for info, base in members(z):
                    parts = info.filename.split("/")
                    run = norm_run(
                        parts[0]
                    )  # trust folder inside zip (Exp-C 'Run-110.zip' holds 'Run 10')
                    sub = "_".join(clean(p) for p in parts[1:-1])
                    name = f"{sub}_{base}" if sub else base
                    w.put(
                        "photos",
                        exp,
                        run,
                        name,
                        z.read(info),
                        rel,
                        info.filename,
                        info.file_size,
                    )
        else:
            raise RuntimeError(f"unhandled file: {path}")
    w.save()
    print(f"{split}: {len(w.rows)} files -> {w.dir}")


# ---------------------------------------------------------------- high frequency


def flatten_hf(split, dest_root, exp=None, run=None, workers=8):
    share, w = Share(split), Writer(split, dest_root)
    reserve, pending = threading.Lock(), {}
    try:
        for path, size in share.walk(f"{share.root}/high-frequency"):
            name = path.rsplit("/", 1)[1]
            z_exp, z_run = (
                exp_of(name),
                norm_run(re.search(r"(Run-\d+)", name).group(1)),
            )
            if (exp and exp.lower() != z_exp.lower()) or (
                run and norm_run(run) != z_run
            ):
                continue
            rel = path[len(share.root) + 1 :]

            def open_zip():
                return zipfile.ZipFile(
                    io.BufferedReader(
                        RemoteFile(share, path, size), buffer_size=16 * 1024**2
                    )
                )

            todo = list(members(open_zip()))
            need = sum(i.file_size for i, _ in todo)
            print(
                f"{rel}: {len(todo)} files, {need / 1e9:.1f} GB extracted", flush=True
            )

            def extract(item):
                # one remote handle per member: the server caps each stream at ~4 MB/s, so parallelism is the speedup
                info, base = item
                with reserve:
                    free = shutil.disk_usage(dest_root).free - sum(pending.values())
                    if free - info.file_size < MIN_FREE:
                        raise SystemExit(
                            f"stopping: {free / 1e9:.1f} GB free on {dest_root}; use --dest on a larger volume"
                        )
                    pending[info.filename] = info.file_size
                try:
                    with open_zip() as zf, zf.open(info) as src:
                        w.put(
                            "hf",
                            z_exp,
                            z_run,
                            base,
                            src,
                            rel,
                            info.filename,
                            info.file_size,
                        )
                finally:
                    with reserve:
                        pending.pop(info.filename)
                        w.save()  # checkpoint so reruns resume

            pool = ThreadPoolExecutor(workers)
            try:
                for fut in as_completed([pool.submit(extract, t) for t in todo]):
                    fut.result()
            finally:
                pool.shutdown(cancel_futures=True)
    finally:
        w.save()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("mode", choices=["small", "hf"])
    ap.add_argument("--split", choices=list(SHARES), action="append")
    ap.add_argument("--exp")
    ap.add_argument("--run")
    ap.add_argument("--dest", type=Path, default=ROOT)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    for split in a.split or list(SHARES):
        if a.mode == "small":
            flatten_small(split)
        else:
            flatten_hf(split, a.dest, a.exp, a.run, a.workers)


if __name__ == "__main__":
    main()
