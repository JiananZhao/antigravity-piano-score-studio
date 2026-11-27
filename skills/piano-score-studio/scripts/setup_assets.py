"""Environment check, tool auto-install, and asset setup for Piano Score Studio."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

SALAMANDER_URL = "https://github.com/sfzinstruments/SalamanderGrandPianoV3/archive/refs/heads/master.tar.gz"
SFIZZ_LATEST_API = "https://api.github.com/repos/sfztools/sfizz/releases/latest"
MUSCRIPTOR_REPO_ID = "MuScriptor/muscriptor-medium"


def find_plugin_root() -> Path:
    # scripts/ -> piano-score-studio/ -> skills/ -> plugin root
    return Path(__file__).resolve().parents[3]


def resolve_config_path(plugin_root: Path, config_arg: str | None = None) -> Path:
    if config_arg:
        return Path(config_arg)
    return plugin_root / "config.json"


def load_config(plugin_root: Path, config_arg: str | None = None) -> dict:
    cfg_path = resolve_config_path(plugin_root, config_arg)
    if not cfg_path.is_file():
        cfg_path = plugin_root / "config.example.json"
    return json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}


def find_musescore(cfg: dict) -> str | None:
    cand = cfg.get("musescore_exe") or os.environ.get("MUSCRIPTOR_MUSESCORE")
    if cand and Path(cand).is_file():
        return str(Path(cand))
    for name in ("mscore", "musescore", "MuseScore4", "musescore4", "mscore4"):
        hit = shutil.which(name)
        if hit:
            return hit
    platform_candidates = (
        Path(r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe"),
        Path(r"C:\Program Files (x86)\MuseScore 4\bin\MuseScore4.exe"),
        Path.home() / "AppData" / "Local" / "Programs" / "MuseScore 4" / "bin" / "MuseScore4.exe",
        Path("/Applications/MuseScore 4.app/Contents/MacOS/mscore"),
        Path("/usr/bin/mscore"),
        Path("/usr/bin/musescore"),
        Path("/usr/local/bin/mscore"),
        Path("/usr/local/bin/musescore"),
    )
    for p in platform_candidates:
        if p.is_file():
            return str(p)
    return None


def find_local_sfizz(plugin_root: Path) -> str | None:
    sfizz_dir = plugin_root / "assets_local" / "sfizz"
    if not sfizz_dir.is_dir():
        return None
    names = ("sfizz_render.exe", "sfizz_render") if sys.platform == "win32" else ("sfizz_render",)
    for name in names:
        for hit in sfizz_dir.rglob(name):
            if hit.is_file():
                return str(hit)
    return None


def find_renderer(cfg: dict, plugin_root: Path) -> dict:
    sfizz_cfg = cfg.get("sfizz_render_exe")
    sfizz: str | None = None
    if sfizz_cfg:
        if Path(str(sfizz_cfg)).is_file():
            sfizz = str(Path(str(sfizz_cfg)))
        else:
            sfizz = shutil.which(str(sfizz_cfg))
    if not sfizz:
        sfizz = shutil.which("sfizz_render") or find_local_sfizz(plugin_root)

    ml_root_str = cfg.get("music_learning_root") or os.environ.get("MUSIC_LEARNING_ROOT", "")
    native_ok = bool(ml_root_str) and (Path(ml_root_str) / "scripts" / "native_host.py").is_file()
    return {
        "sfizz_render": sfizz,
        "music_learning_native_host": ml_root_str if native_ok else None,
    }


def find_salamander_sfz(plugin_root: Path, cfg: dict) -> Path | None:
    rel = cfg.get("salamander_sfz", "assets_local/salamander/Salamander Grand Piano V3.sfz")
    candidates = [
        Path(rel) if Path(rel).is_absolute() else (plugin_root / rel),
        plugin_root / "assets_local" / "salamander" / "Salamander Grand Piano V3.sfz",
    ]
    ml_root = cfg.get("music_learning_root") or os.environ.get("MUSIC_LEARNING_ROOT", "")
    if ml_root:
        candidates.append(Path(ml_root) / "data" / "instruments" / "render_tmp" / "resources" / "Salamander Grand Piano V3.sfz")
    for p in candidates:
        if p.is_file():
            return p
    sal_dir = plugin_root / "assets_local" / "salamander"
    if sal_dir.is_dir():
        for sfz in sorted(sal_dir.rglob("*Salamander*.sfz")):
            if sfz.is_file():
                return sfz
    return None


def resolve_muscriptor_dir(plugin_root: Path, cfg: dict) -> Path:
    rel = cfg.get("muscriptor_model_dir") or "models/medium"
    p = Path(rel)
    return p if p.is_absolute() else (plugin_root / p)


def find_muscriptor_model(plugin_root: Path, cfg: dict) -> tuple[Path, bool]:
    model_dir = resolve_muscriptor_dir(plugin_root, cfg)
    ready = (model_dir / "config.json").is_file() and (model_dir / "model.safetensors").is_file()
    return model_dir, ready


def download_salamander(dest_dir: Path, plugin_root: Path, cfg: dict) -> Path | None:
    existing = find_salamander_sfz(plugin_root, cfg)
    if existing:
        print(f"Salamander Grand Piano V3 already available: {existing}")
        return existing
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive = dest_dir / "salamander_v3.tar.gz"
    print(f"Downloading Salamander Grand Piano V3 (CC-BY 3.0, ~714 MB) from {SALAMANDER_URL} ...")
    urllib.request.urlretrieve(SALAMANDER_URL, str(archive))
    print("Extracting archive...")
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(dest_dir)
    archive.unlink(missing_ok=True)
    for sfz in sorted(dest_dir.rglob("*Salamander*.sfz")):
        if sfz.is_file():
            print(f"Ready: {sfz}")
            return sfz
    raise FileNotFoundError("Extracted Salamander archive but could not locate .sfz file.")


def _select_sfizz_asset(assets: list[dict]) -> dict | None:
    if sys.platform == "win32":
        zips = [a for a in assets if a.get("name", "").lower().endswith(".zip") and "win" in a.get("name", "").lower()]
        for pref in ("win64", "x64", "amd64", "win"):
            for a in zips:
                if pref in a.get("name", "").lower():
                    return a
        return zips[0] if zips else None
    if sys.platform.startswith("linux"):
        tars = [
            a for a in assets
            if any(a.get("name", "").lower().endswith(ext) for ext in (".tar.gz", ".tgz", ".tar.xz"))
            and "linux" in a.get("name", "").lower()
        ]
        return tars[0] if tars else None
    return None


def download_sfizz(dest_dir: Path, cfg: dict, plugin_root: Path) -> Path | None:
    existing = find_renderer(cfg, plugin_root).get("sfizz_render")
    if existing:
        print(f"sfizz_render already available: {existing}")
        return Path(existing)

    if sys.platform == "darwin":
        print("macOS detected: install sfizz via Homebrew (`brew install sfizz`) or https://sfztools.github.io/sfizz/downloads/")
        return None

    try:
        req = urllib.request.Request(SFIZZ_LATEST_API, headers={"User-Agent": "piano-score-studio-installer"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            release = json.loads(resp.read().decode("utf-8"))
        asset = _select_sfizz_asset(release.get("assets", []))
        if not asset:
            print("Warning: No matching prebuilt sfizz archive found for this OS. Please install from https://sfztools.github.io/sfizz/downloads/")
            return None

        dest_dir.mkdir(parents=True, exist_ok=True)
        pkg_name = asset["name"]
        url = asset["browser_download_url"]
        archive = dest_dir / pkg_name
        print(f"Downloading sfizz ({pkg_name}) from {url} ...")
        urllib.request.urlretrieve(url, str(archive))
        print("Extracting sfizz archive...")
        if pkg_name.lower().endswith(".zip"):
            with zipfile.ZipFile(archive, "r") as zf:
                zf.extractall(dest_dir)
        else:
            with tarfile.open(archive, "r:*") as tf:
                tf.extractall(dest_dir)
        archive.unlink(missing_ok=True)

        hit = find_local_sfizz(plugin_root)
        if hit:
            p = Path(hit)
            if sys.platform != "win32":
                p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            print(f"Ready: {p}")
            return p
        print("Warning: Extracted sfizz archive, but 'sfizz_render' executable was not found inside assets_local/sfizz/.")
        print("Download manually from https://sfztools.github.io/sfizz/downloads/ if needed.")
        return None
    except Exception as exc:
        print(f"Warning: Could not auto-download sfizz_render ({exc}).")
        print("You can install it manually from https://sfztools.github.io/sfizz/downloads/ or use music-learning NativeHost.")
        return None


def download_muscriptor(dest_dir: Path) -> bool:
    if (dest_dir / "config.json").is_file() and (dest_dir / "model.safetensors").is_file():
        print(f"MuScriptor model already ready: {dest_dir}")
        return True

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Warning: huggingface_hub is not installed. Run `pip install -r requirements.txt` first.")
        return False

    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {MUSCRIPTOR_REPO_ID} (~1.17 GB, CC BY-NC 4.0) to {dest_dir} ...")
    try:
        for fname in ("config.json", "model.safetensors"):
            hf_hub_download(repo_id=MUSCRIPTOR_REPO_ID, filename=fname, local_dir=str(dest_dir))
        print(f"Ready: {dest_dir}")
        return True
    except Exception as exc:
        print(f"\n[MuScriptor Download Action Required] Could not download {MUSCRIPTOR_REPO_ID} automatically ({exc}).")
        print("HuggingFace requires each user to accept the CC BY-NC 4.0 gated license once:")
        print("  1. Open https://huggingface.co/MuScriptor/muscriptor-medium and click 'Agree and access repository'.")
        print("  2. Run `huggingface-cli login` (or set the HF_TOKEN environment variable).")
        print("  3. Re-run `python skills/piano-score-studio/scripts/setup_assets.py --download-muscriptor --write-config`.\n")
        return False


def _format_cfg_path(path_str: str, plugin_root: Path) -> str:
    p = Path(path_str)
    try:
        rel = p.resolve().relative_to(plugin_root.resolve())
        return rel.as_posix()
    except Exception:
        return str(p)


def write_config(plugin_root: Path, cfg_arg: str | None, status: dict) -> Path:
    cfg_path = resolve_config_path(plugin_root, cfg_arg)
    cfg = load_config(plugin_root, cfg_arg)
    if status.get("musescore4"):
        cfg["musescore_exe"] = status["musescore4"]
    renderers = status.get("renderers") or {}
    if renderers.get("sfizz_render"):
        cfg["sfizz_render_exe"] = renderers["sfizz_render"]
    if renderers.get("music_learning_native_host"):
        cfg["music_learning_root"] = renderers["music_learning_native_host"]
    if status.get("salamander_sfz"):
        cfg["salamander_sfz"] = _format_cfg_path(status["salamander_sfz"], plugin_root)
    if status.get("muscriptor_model_dir"):
        cfg["muscriptor_model_dir"] = _format_cfg_path(status["muscriptor_model_dir"], plugin_root)

    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return cfg_path


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="Check environment and optionally install tools/assets for Piano Score Studio")
    ap.add_argument("--config", default=None)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--download-sfizz", action="store_true", help="Auto-download official sfizz_render binary into assets_local/sfizz/")
    ap.add_argument("--download-salamander", action="store_true", help="Auto-download Salamander Grand Piano V3 (CC-BY 3.0, ~714 MB)")
    ap.add_argument("--download-muscriptor", action="store_true", help="Download MuScriptor/muscriptor-medium (~1.17 GB) using your HF login/token")
    ap.add_argument("--write-config", action="store_true", help="Persist detected paths into config.json")
    args = ap.parse_args()

    plugin_root = find_plugin_root()
    cfg = load_config(plugin_root, args.config)

    if args.download_sfizz:
        download_sfizz(plugin_root / "assets_local" / "sfizz", cfg, plugin_root)
    if args.download_salamander:
        download_salamander(plugin_root / "assets_local" / "salamander", plugin_root, cfg)
    if args.download_muscriptor:
        download_muscriptor(resolve_muscriptor_dir(plugin_root, cfg))

    muscriptor_dir, muscriptor_ready = find_muscriptor_model(plugin_root, cfg)
    status: dict = {
        "plugin_root": str(plugin_root),
        "python": sys.version.split()[0],
        "musescore4": find_musescore(cfg),
        "renderers": find_renderer(cfg, plugin_root),
        "salamander_sfz": str(find_salamander_sfz(plugin_root, cfg) or ""),
        "muscriptor_model_dir": str(muscriptor_dir),
        "muscriptor_model_ready": muscriptor_ready,
    }
    for pkg in ("numpy", "soundfile", "mido", "torch", "muscriptor", "beat_this"):
        try:
            mod = __import__(pkg)
            status[f"pkg_{pkg}"] = True
            if pkg == "torch":
                status["cuda_available"] = bool(mod.cuda.is_available())
        except ImportError:
            status[f"pkg_{pkg}"] = False
            if pkg == "torch":
                status["cuda_available"] = False

    if args.write_config:
        cfg_written = write_config(plugin_root, args.config, status)
        status["config_written"] = str(cfg_written)

    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
