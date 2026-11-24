"""Download / locate model weights (FER weights from your GitHub Release, sherpa-onnx models, YuNet)."""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

from hri.config import Config

log = logging.getLogger(__name__)


def sha256sum(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def find_file(folder: Path, pattern: str, prefer: str = "int8") -> Path:
    """Glob inside a model folder; prefer files containing `prefer` (e.g. int8) if any."""
    hits = sorted(folder.rglob(pattern))
    if not hits:
        raise FileNotFoundError(f"No file matching {pattern!r} in {folder}")
    preferred = [h for h in hits if prefer and prefer in h.name]
    return preferred[0] if preferred else hits[0]


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "hri-robot-downloader"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while block := r.read(1 << 16):
            f.write(block)
            done += len(block)
            if total and sys.stderr.isatty():
                print(f"\r  {done / 1e6:7.1f}/{total / 1e6:.1f} MB", end="", file=sys.stderr)
    if total and sys.stderr.isatty():
        print(file=sys.stderr)
    tmp.replace(dest)


def _check_member(root: Path, dest: Path, name: str) -> None:
    target = (dest / name).resolve()
    if root not in target.parents and target != root:
        raise RuntimeError(f"Unsafe path in archive: {name}")


def _safe_extract(archive: Path, dest: Path) -> None:
    """Extract .tar.* or .zip (.ferpack) into dest, rejecting path traversal and links."""
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                _check_member(root, dest, name)
            z.extractall(dest)
        return
    with tarfile.open(archive) as tar:
        for m in tar.getmembers():
            _check_member(root, dest, m.name)
            if m.issym() or m.islnk():
                raise RuntimeError(f"Links not allowed in archive: {m.name}")
        tar.extractall(dest)


class ModelRegistry:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.dir = cfg.models_dir
        self.manifest_path = cfg.path("models.manifest")
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))["models"]

    # ---------- lookup ----------
    def spec(self, name: str) -> dict:
        if name not in self.manifest:
            raise KeyError(f"Model {name!r} not in {self.manifest_path}")
        return self.manifest[name]

    def url(self, name: str) -> str:
        rel = self.cfg.section("models.release")
        return self.spec(name)["url"].format(release_repo=rel.get("repo", ""), release_tag=rel.get("tag", ""))

    def path(self, name: str) -> Path:
        """File path (single-file model) or folder path (archive model)."""
        s = self.spec(name)
        return self.dir / (s["dirname"] if s.get("archive") else s["filename"])

    def is_present(self, name: str) -> bool:
        return self.path(name).exists()

    def require(self, name: str) -> Path:
        p = self.path(name)
        if not p.exists():
            raise FileNotFoundError(
                f"Model {name!r} missing at {p}. Run: hri models download {name}")
        return p

    # ---------- download ----------
    def download(self, name: str, force: bool = False) -> Path:
        s = self.spec(name)
        target = self.path(name)
        if target.exists() and not force:
            log.info("%s: already present (%s)", name, target)
            return target
        url = self.url(name)
        log.info("%s: downloading %s", name, url)
        if s.get("archive"):
            arc = self.dir / Path(url).name
            _download(url, arc)
            self._verify(name, arc, s)
            if target.exists():
                shutil.rmtree(target)
            # "zip" packs (.ferpack) are flat -> extract into their own folder;
            # sherpa-onnx tarballs already contain a top-level folder
            _safe_extract(arc, target if s["archive"] == "zip" else self.dir)
            arc.unlink()
        else:
            _download(url, target)
            self._verify(name, target, s)
        log.info("%s: ok -> %s", name, target)
        return target

    @staticmethod
    def _verify(name: str, file: Path, s: dict) -> None:
        expected = s.get("sha256")
        if not expected:
            log.warning("%s: no sha256 in manifest, skipping verification", name)
            return
        actual = sha256sum(file)
        if actual != expected:
            file.unlink(missing_ok=True)
            raise RuntimeError(f"{name}: sha256 mismatch (expected {expected}, got {actual})")

    # ---------- publish FER weights to GitHub Release ----------
    def publish(self, file: Path, name: str, tag: str | None = None, repo: str | None = None) -> str:
        """Upload `file` to a GitHub Release with `gh` and write its sha256 into the manifest."""
        if shutil.which("gh") is None:
            raise RuntimeError("GitHub CLI `gh` not found (https://cli.github.com)")
        rel = self.cfg.section("models.release")
        tag = tag or rel["tag"]
        repo = repo or rel["repo"]
        spec = self.spec(name)
        if spec.get("archive"):
            raise ValueError("publish supports single-file models only")
        if file.name != spec["filename"]:
            raise ValueError(f"File must be named {spec['filename']!r} to match the manifest")
        digest = sha256sum(file)
        exists = subprocess.run(["gh", "release", "view", tag, "-R", repo],
                                capture_output=True).returncode == 0
        if not exists:
            subprocess.run(["gh", "release", "create", tag, "-R", repo, "-t", tag,
                            "-n", "Model weights for hri-robot"], check=True)
        subprocess.run(["gh", "release", "upload", tag, str(file), "-R", repo, "--clobber"], check=True)

        raw = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        raw["models"][name]["sha256"] = digest
        self.manifest_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        self.manifest = raw["models"]
        return digest
