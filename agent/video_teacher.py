"""Learn reusable Blender modeling concepts from YouTube or local videos."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import numpy as np

from agent import knowledge as kb
from agent.abstract import collect_params, infer_ops_from_actions, new_op
from agent.paths import ensure_dataset, videos_dir
from agent.policy import PolicyError, chat_text_json, chat_vision_json, parse_action
from agent.window import downsample_for_vlm, encode_jpeg_b64

ProgressFn = Callable[[str], None]

_URL_RE = re.compile(r"^https?://", re.I)
_VIDEO_SUFFIXES = {".mp4", ".webm", ".mkv", ".avi", ".mov", ".m4v"}
MAX_LOCAL_VIDEO_BYTES = 4_000_000_000
MAX_CAPTION_BYTES = 8_000_000
MAX_KEYFRAME_SIDE = 640
_YOUTUBE_HOSTS = {
    "youtu.be",
    "www.youtu.be",
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
}
_FILLER = re.compile(
    r"\b(subscribe|like and subscribe|patreon|sponsor|outro|intro music|thanks for watching)\b",
    re.I,
)

_KNOWN_NAMES = ", ".join(
    sorted({s.label.lower() for s in kb.SPECS.values() if not s.op.startswith("mesh.primitive_")})
)

_EXTRA_TECHNIQUES = (
    "boolean, knife, bridge edge loops, solidify, array, mirror, subdivision surface, "
    "shrinkwrap, bevel modifier, join, separate, duplicate, merge by distance, grid fill, "
    "bisect, spin, screw, dissolve, poke, wireframe, shade smooth, proportional editing, "
    "snap, parenting, empty, curve, remesh, decimate, displace, lattice, hook"
)

CONCEPT_LIST_SYSTEM = f"""Blender modeling teacher. Extract DISTINCT transferable CONCEPTS (techniques), not a mouse screenplay.
JSON only: {{"concepts":[{{"name":"bridge edge loops","summary":"...","when_to_use":"...","preconditions":["edit mode","two edge loops selected"],"aliases":["bridge"],"params":{{}},"t_start":12.0,"t_end":40.0}}]}}
Rules:
- Prefer SPECIFIC lesson names (boolean difference, knife project, solidify, array modifier, bridge edge loops, loop cut for windows, extrude roof…). Do NOT collapse everything into only extrude/inset/bevel/move/scale/loop cut.
- Known ops you may use as names when accurate: {_KNOWN_NAMES}. Also fine: {_EXTRA_TECHNIQUES}.
- Each concept must be a different technique or clearly different use (e.g. "extrude cabin" vs "extrude"). Max 32 concepts per segment.
- Plain-language summary; preconditions for the USER mesh; numeric values go in params; skip subscribe/outro filler.
"""

CONCEPT_STEPS_SYSTEM = """Turn one Blender CONCEPT into keyboard actions for the user's mesh.
JSON only: {"steps":[{"action":"key|hotkey|type|wait","keys":[],"text":"","seconds":0.4,"reason":"..."}]}
Prefer real Blender hotkeys (Tab, E, I, Ctrl+B, Ctrl+R, K, F, P, Ctrl+J, G/S/R + number + Enter) or F3 + operator name + Enter.
NEVER invent click x/y from the video. Max 10 steps. No stop.
"""

# Caption / narration hints → seed concepts when the VLM only returns the basics.
_CAPTION_TECHNIQUES: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"\bboolean\b", re.I), "boolean", "Boolean difference/union on meshes"),
    (re.compile(r"\bknife(\s+project)?\b", re.I), "knife", "Cut topology with the knife tool"),
    (re.compile(r"\bbridge\b", re.I), "bridge edge loops", "Connect two edge loops"),
    (re.compile(r"\bsolidify\b", re.I), "solidify", "Add thickness to faces or a solidify modifier"),
    (re.compile(r"\barray(\s+modifier)?\b", re.I), "array modifier", "Repeat geometry with an Array modifier"),
    (re.compile(r"\bmirror(\s+modifier)?\b", re.I), "mirror modifier", "Symmetry with a Mirror modifier"),
    (re.compile(r"\bsubdiv(ision)?(\s+surface|\s+modifier)?\b", re.I), "subdivision surface", "Smooth with Subdivision Surface"),
    (re.compile(r"\bshrink\s*wrap\b", re.I), "shrinkwrap", "Project mesh onto another with Shrinkwrap"),
    (re.compile(r"\bgrid\s*fill\b", re.I), "grid fill", "Fill a hole with a grid of quads"),
    (re.compile(r"\bbisect\b", re.I), "bisect", "Cut the mesh with a plane"),
    (re.compile(r"\b(spin|lathe)\b", re.I), "spin", "Spin geometry around the cursor"),
    (re.compile(r"\bscrew\b", re.I), "screw", "Screw / revolve geometry"),
    (re.compile(r"\bdissolve\b", re.I), "dissolve", "Dissolve edges or faces"),
    (re.compile(r"\bseparat(e|ion)\b", re.I), "separate", "Separate selection into a new object"),
    (re.compile(r"\bjoin(ed|ing)?\b", re.I), "join", "Join objects into one"),
    (re.compile(r"\bmerge\s+by\s+distance\b|\bremove\s+doubles\b", re.I), "merge by distance", "Weld nearby vertices"),
    (re.compile(r"\bshade\s+smooth\b", re.I), "shade smooth", "Smooth shading"),
    (re.compile(r"\bproportional\s+edit", re.I), "proportional editing", "Falloff editing with O key"),
    (re.compile(r"\bloop\s*cut", re.I), "loop cut", "Add edge loops across a mesh"),
    (re.compile(r"\bbevel\b", re.I), "bevel", "Bevel edges or vertices"),
    (re.compile(r"\binset\b", re.I), "inset", "Inset faces"),
    (re.compile(r"\bextrud", re.I), "extrude", "Extrude selected geometry"),
    (re.compile(r"\b(material|shader|principled|base\s*color)\b", re.I), "material", "Assign a material / base color"),
    (re.compile(r"\bpaint\b", re.I), "texture paint", "Paint color onto the mesh"),
]


def _yt_dlp_cmd() -> list[str]:
    """Prefer the yt-dlp installed in this interpreter's environment (venv Scripts is not on PATH)."""
    try:
        import yt_dlp  # noqa: F401

        return [sys.executable, "-m", "yt_dlp"]
    except ImportError:
        pass
    exe = shutil.which("yt-dlp")
    if exe:
        return [exe]
    return [sys.executable, "-m", "yt_dlp"]


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _progress(cb: ProgressFn | None, msg: str) -> None:
    if cb:
        try:
            cb(msg)
        except Exception:
            pass


def _is_url(raw: str) -> bool:
    return bool(_URL_RE.match((raw or "").strip()))


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def _norm_name(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip().lower())


def _js_runtime_args() -> list[str]:
    """YouTube extraction needs a JS runtime (deno/node) on current yt-dlp builds."""
    args: list[str] = []
    for name in ("deno", "node"):
        path = shutil.which(name)
        if path:
            args += ["--js-runtimes", f"{name}:{path}" if name == "deno" else name]
    if any(a.startswith("deno") or a == "deno" for a in args[1::2]):
        return args
    # WinGet often installs deno off PATH for already-running processes.
    home = Path.home()
    candidates = [
        home / "AppData" / "Local" / "Microsoft" / "WinGet" / "Links" / "deno.exe",
        *sorted(
            (home / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages").glob(
                "DenoLand.Deno*/deno.exe"
            )
        ),
    ]
    for candidate in candidates:
        if candidate.is_file():
            args += ["--js-runtimes", f"deno:{candidate}"]
            break
    return args


def _yt_dlp_base(out_tmpl: str, *, ffmpeg: bool | None = None, subs: bool = True) -> list[str]:
    ffmpeg = _ffmpeg_available() if ffmpeg is None else ffmpeg
    cmd = _yt_dlp_cmd() + [
        "--ignore-config",
        "--no-playlist",
        "--max-filesize",
        "4G",
        # Prefer android/web clients when default player is blocked.
        "--extractor-args",
        "youtube:player_client=android,web",
    ]
    if subs:
        # Subtitle fetch is best-effort — YouTube often 429s auto-captions.
        cmd += [
            "--write-auto-sub",
            "--write-sub",
            "--sub-langs",
            "en.*,en",
            "--sub-format",
            "vtt/srt/best",
            # NOTE: there is no "--ignore-no-subtitles" option; yt-dlp already
            # tolerates missing subs. (The bogus flag aborted every caption run.)
        ]
    else:
        cmd += ["--no-write-subs", "--no-write-auto-subs"]
    cmd += _js_runtime_args()
    if ffmpeg:
        if subs:
            cmd += ["--convert-subs", "vtt"]
        cmd += [
            "-f", "bv*[height<=720]+ba/b[height<=720]/best",
            "--merge-output-format", "mp4",
        ]
    else:
        # Without ffmpeg only single-file (progressive) formats can be saved.
        cmd += ["-f", "b[height<=720][ext=mp4]/b[height<=720]/b"]
    cmd += ["-o", out_tmpl]
    return cmd


def _subtitle_fetch_failed(err: str) -> bool:
    low = (err or "").lower()
    return (
        "subtitle" in low
        or "subtitles" in low
        or "429" in low
        or "too many requests" in low
    )


def _friendly_ytdlp_error(raw: str) -> str:
    text = (raw or "").strip()
    low = text.lower()
    if "not available" in low or "unavailable" in low or "private video" in low:
        return (
            "That YouTube video can't be downloaded (private, deleted, or region-blocked). "
            "Paste a public Blender modeling tutorial URL, or use File with a local mp4."
        )
    if "sign in" in low or "confirm your age" in low:
        return (
            "YouTube blocked the download (login/age gate). "
            "Use a different public tutorial, or download the mp4 yourself and pick it under File."
        )
    if "js runtime" in low or "javascript" in low:
        return (
            "YouTube download needs Deno or Node.js installed. "
            "Install Deno, restart Start GUI Agent.bat, then try again."
        )
    if _subtitle_fetch_failed(text):
        return (
            "YouTube rate-limited caption download. "
            "Retry Learn from Video (video-only works), or pick a local mp4 under File."
        )
    return f"yt-dlp failed: {text[:220]}"


def _prune_teacher_videos(current: Path, dataset_root: Path | None, keep: int = 2) -> None:
    """Delete downloaded tutorials except the newest `keep` (never touches Recorded demos)."""
    try:
        teacher_dir = videos_dir(ensure_dataset(dataset_root)) / "teacher"
        if not teacher_dir.is_dir():
            return
        try:
            if current.resolve().parent != teacher_dir.resolve():
                # A local file the user picked, or a Recorded demo: keep it.
                current = teacher_dir / "__none__"
        except OSError:
            return
        files = sorted(
            (p for p in teacher_dir.iterdir() if p.is_file()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        keep_stems = {current.stem}
        for p in files:
            if len(keep_stems) >= keep:
                break
            if p.suffix.lower() in {".mp4", ".webm", ".mkv", ".avi"}:
                keep_stems.add(p.stem)
        for p in files:
            stem = p.stem.split(".")[0]  # "abc.en.vtt" → "abc"
            if stem in keep_stems and not p.name.endswith(".part"):
                continue
            try:
                p.unlink()
            except OSError:
                pass
    except Exception:
        pass


def _last_local_demo_video(dataset_root: Path | None) -> Path | None:
    """Prefer the user's last Record session mp4 when YouTube is unusable."""
    root = ensure_dataset(dataset_root)
    meta_path = root / "last_session.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            meta = {}
        vid_name = str(meta.get("video") or "").strip()
        if vid_name:
            candidate = (videos_dir(root) / Path(vid_name).name).resolve()
            if candidate.is_file():
                return candidate
        sid = str(meta.get("session_id") or "").strip()
        if sid:
            for ext in (".mp4", ".webm", ".mkv"):
                candidate = videos_dir(root) / f"{sid}{ext}"
                if candidate.is_file():
                    return candidate
    videos = sorted(
        (
            p
            for p in videos_dir(root).glob("*")
            if p.is_file() and p.suffix.lower() in {".mp4", ".webm", ".mkv", ".avi"}
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return videos[0] if videos else None


def merge_concepts(concepts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Dedupe by name; keep richer summary/aliases/time span."""
    by_key: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for raw in concepts:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name") or "").strip()
        if not name or _FILLER.search(name):
            continue
        key = _norm_name(name)
        if key in {"intro", "outro", "subscribe"}:
            continue
        aliases = [str(a).strip() for a in (raw.get("aliases") or []) if str(a).strip()]
        if isinstance(raw.get("aliases"), str):
            aliases = [p.strip() for p in str(raw["aliases"]).split(",") if p.strip()]
        card = {
            "name": name,
            "summary": str(raw.get("summary") or "").strip(),
            "when_to_use": str(raw.get("when_to_use") or "").strip(),
            "preconditions": raw.get("preconditions") or [],
            "aliases": aliases,
            "t_start": float(raw.get("t_start") or 0.0),
            "t_end": float(raw.get("t_end") or 0.0),
            "steps": [s for s in (raw.get("steps") or []) if isinstance(s, dict)],
            "ops": [o for o in (raw.get("ops") or []) if isinstance(o, dict)],
            "params": dict(raw.get("params") or {}) if isinstance(raw.get("params"), dict) else {},
            "video": str(raw.get("video") or ""),
        }
        if key not in by_key:
            by_key[key] = card
            order.append(key)
            continue
        old = by_key[key]
        if len(card["summary"]) > len(str(old.get("summary") or "")):
            old["summary"] = card["summary"]
        if len(card["when_to_use"]) > len(str(old.get("when_to_use") or "")):
            old["when_to_use"] = card["when_to_use"]
        seen_a = {a.lower() for a in old.get("aliases") or []}
        for a in aliases:
            if a.lower() not in seen_a:
                old.setdefault("aliases", []).append(a)
                seen_a.add(a.lower())
        old["t_start"] = min(float(old.get("t_start") or 0), card["t_start"])
        old["t_end"] = max(float(old.get("t_end") or 0), card["t_end"])
        if len(card["steps"]) > len(old.get("steps") or []):
            old["steps"] = card["steps"]
        if len(card["ops"]) > len(old.get("ops") or []):
            old["ops"] = card["ops"]
        merged_params = dict(old.get("params") or {})
        merged_params.update(card["params"])
        old["params"] = merged_params
        if card.get("video") and not old.get("video"):
            old["video"] = card["video"]
    return [by_key[k] for k in order]


def ground_concept(concept: dict[str, Any]) -> dict[str, Any]:
    """Attach semantic ops to a concept so it becomes a parameterizable skill.

    Priority: knowledge-base op matching the concept name/aliases (most reliable),
    then ops inferred from the LLM's hotkey steps, then an F3 search of the name.
    """
    out = dict(concept)
    name = str(out.get("name") or "")
    labels = [name, *[str(a) for a in (out.get("aliases") or [])]]
    kb_op = None
    for label in labels:
        kb_op = kb.op_for_phrase(label)
        if kb_op:
            break
    params_hint = dict(out.get("params") or {}) if isinstance(out.get("params"), dict) else {}
    pre = " ".join(str(p) for p in (out.get("preconditions") or [])).lower()
    start_mode = "EDIT" if "edit" in pre or "face" in pre or "vert" in pre or "edge" in pre else "OBJECT"
    inferred = infer_ops_from_actions(list(out.get("steps") or []), start_mode=start_mode)
    ops: list[dict[str, Any]] = []
    if kb_op:
        spec = kb.get(kb_op)
        params: dict[str, Any] = {}
        for pname in (spec.params if spec else ()):
            val = params_hint.get(pname)
            if val is None:
                continue
            try:
                if pname in {"axis", "type", "action", "target", "color", "name"}:
                    params[pname] = str(val)
                elif pname in {"segments", "number_cuts", "level"}:
                    params[pname] = int(float(val))
                else:
                    params[pname] = float(val)
            except (TypeError, ValueError):
                continue
        # Keep the LLM's numeric value if it typed one for the same op.
        for row in inferred:
            if row.get("op") == kb_op:
                for k, v in (row.get("params") or {}).items():
                    params.setdefault(k, v)
        if kb_op == "material.set" and "color" not in params:
            cname = kb.color_name_in_text(" ".join(labels))
            if cname:
                params["color"] = cname
        ops = [new_op(kb_op, params)]
        extra = [r for r in inferred if r.get("op") not in {kb_op, "key", "search", "object.editmode_toggle"}]
        ops.extend(extra[:4])
    elif inferred and any(kb.get(str(r.get("op") or "")) for r in inferred):
        ops = [r for r in inferred if r.get("op") != "object.editmode_toggle"][:6]
    elif name:
        from agent.planner import is_complex_object_goal

        # F3-searching "Make Me An Advanced House" is not a technique.
        if not is_complex_object_goal(name):
            ops = [new_op("search", {"text": name.title()})]
    out["ops"] = ops
    out["params"] = collect_params(ops)
    out["start_mode"] = start_mode
    return out


def sanitize_concept_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep transferable Hands steps; drop coordinate clicks from video frames."""
    out: list[dict[str, Any]] = []
    for raw in steps:
        if not isinstance(raw, dict):
            continue
        try:
            action = parse_action(json.dumps(raw))
        except PolicyError:
            kind = str(raw.get("action") or "").lower()
            if kind not in {"key", "hotkey", "type", "wait"}:
                continue
            action = {
                "action": kind,
                "keys": raw.get("keys") or [],
                "text": str(raw.get("text") or "")[:64],
                "seconds": float(raw.get("seconds") or 0.4),
                "reason": str(raw.get("reason") or "")[:120],
            }
        kind = str(action.get("action") or "")
        if kind in {"click", "drag"}:
            # Video pixel coords do not transfer; skip.
            continue
        if kind == "stop":
            continue
        if kind == "wait":
            sec = float(action.get("seconds") or raw.get("seconds") or 0.4)
            out.append(
                {
                    "action": "wait",
                    "seconds": float(max(0.15, min(2.0, sec))),
                    "reason": str(action.get("reason") or "concept pause"),
                }
            )
            continue
        step: dict[str, Any] = {
            "action": kind,
            "keys": list(action.get("keys") or [])[:3],
            "text": str(action.get("text") or "")[:64],
            "reason": str(action.get("reason") or "concept step")[:120],
        }
        out.append(step)
        # Ensure menu/search breathing room.
        if kind == "key" and [str(k).lower() for k in step["keys"]] == ["f3"]:
            out.append({"action": "wait", "seconds": 0.45, "reason": "wait for operator search"})
        if kind == "type":
            out.append({"action": "wait", "seconds": 0.35, "reason": "wait for search results"})
    # Cap length.
    return out[:12]


def _read_caption_file(path: Path) -> list[tuple[float, str]]:
    """Return [(t_seconds, text)] from vtt/srt/txt."""
    if not path.is_file():
        return []
    try:
        if path.stat().st_size > MAX_CAPTION_BYTES:
            return []
    except OSError:
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    cues: list[tuple[float, str]] = []
    # WebVTT / SRT timestamps
    ts_re = re.compile(
        r"(\d{1,2}:)?(\d{1,2}):(\d{2})[.,](\d{1,3})\s*-->\s*(\d{1,2}:)?(\d{1,2}):(\d{2})[.,](\d{1,3})"
    )

    def _to_sec(h: str | None, m: str, s: str, ms: str) -> float:
        hours = int(h[:-1]) if h else 0
        return hours * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")[:3]) / 1000.0

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m = ts_re.search(line)
        if m:
            t0 = _to_sec(m.group(1), m.group(2), m.group(3), m.group(4))
            body: list[str] = []
            i += 1
            while i < len(lines) and lines[i].strip():
                body.append(re.sub(r"<[^>]+>", "", lines[i]).strip())
                i += 1
            joined = " ".join(b for b in body if b)
            if joined and not _FILLER.search(joined):
                cues.append((t0, joined))
            continue
        i += 1
    if cues:
        return cues
    # Plain transcript fallback.
    for i, line in enumerate(lines):
        line = line.strip()
        if line and not line.startswith("WEBVTT"):
            cues.append((float(i), line))
    return cues


def captions_window(cues: list[tuple[float, str]], t0: float, t1: float, *, limit: int = 40) -> str:
    bits = [txt for t, txt in cues if t0 - 1.0 <= t <= t1 + 1.0]
    if not bits and cues:
        bits = [txt for _, txt in cues[:limit]]
    return " ".join(bits[:limit])[:2500]


def _is_youtube_url(link: str) -> bool:
    parsed = urlparse(link)
    if parsed.scheme not in {"http", "https"}:
        return False
    if parsed.username or parsed.password:
        return False
    host = (parsed.hostname or "").lower().rstrip(".")
    if host in _YOUTUBE_HOSTS:
        return True
    return host.endswith(".youtube.com") and host.count(".") >= 2


def _clean_video_url(raw: str) -> str:
    """Normalize YouTube links (drop playlist/time junk that breaks some extractors)."""
    link = (raw or "").strip()
    if not link:
        return ""
    # youtu.be/ID?t=315 → keep id only
    m = re.match(r"https?://(?:www\.)?youtu\.be/([\w-]{6,})", link, re.I)
    if m:
        return f"https://www.youtube.com/watch?v={m.group(1)}"
    m = re.search(r"[?&]v=([\w-]{6,})", link)
    if m:
        return f"https://www.youtube.com/watch?v={m.group(1)}"
    return link.split("&")[0].split("#")[0]


def ensure_video_path_in_dataset(path: str, dataset_root: Path | None) -> Path | None:
    """Return a video file under the agent dataset, or None if that path is missing.

    Paths that resolve outside the dataset raise ValueError before the file is
    opened. /video/learn must not read an arbitrary video on the machine.
    """
    text = str(path or "").strip()
    if not text:
        return None
    root = ensure_dataset(dataset_root)
    local = Path(text).expanduser()
    try:
        resolved = local.resolve()
        base = root.resolve()
        resolved.relative_to(base)
    except ValueError as exc:
        raise ValueError("Local video must be inside the agent dataset folder") from exc
    except OSError as exc:
        raise ValueError("Video file path is not readable") from exc
    if not resolved.exists():
        return None
    if not resolved.is_file():
        raise ValueError("Video file path is not readable")
    if resolved.suffix.lower() not in _VIDEO_SUFFIXES:
        raise ValueError("Local video must be mp4, webm, mkv, avi, mov, or m4v")
    try:
        size = resolved.stat().st_size
    except OSError as exc:
        raise ValueError("Video file path is not readable") from exc
    if size > MAX_LOCAL_VIDEO_BYTES:
        raise ValueError("Local video is too large (over 4 GB)")
    return resolved


def _caption_under_dataset(path: Path, root: Path) -> Path | None:
    try:
        resolved = path.resolve()
        resolved.relative_to(root.resolve())
    except (OSError, ValueError):
        return None
    if resolved.is_file():
        return resolved
    return None


def fetch_video_source(
    *,
    url: str = "",
    path: str = "",
    dataset_root: Path | None = None,
    max_minutes: float = 6.0,
    on_progress: ProgressFn | None = None,
) -> tuple[Path, list[tuple[float, str]], str]:
    """Return (video_path, caption_cues, title). Downloads YouTube via yt-dlp when needed."""
    root = ensure_dataset(dataset_root)
    dest_dir = videos_dir(root) / "teacher"
    dest_dir.mkdir(parents=True, exist_ok=True)
    max_minutes = float(max(1.0, min(1200.0, max_minutes)))

    resolved = ensure_video_path_in_dataset(path, root)
    if resolved is not None:
        _progress(on_progress, f"Using local video {resolved.name}")
        cues: list[tuple[float, str]] = []
        for ext in (".vtt", ".srt", ".txt"):
            side = _caption_under_dataset(resolved.with_suffix(ext), root)
            if side is not None:
                cues = _read_caption_file(side)
                break
        return resolved, cues, resolved.stem

    link = _clean_video_url(url)
    if not link or not _is_url(link) or not _is_youtube_url(link):
        raise ValueError("Provide a YouTube URL or a local video file path")

    session = uuid.uuid4().hex[:10]
    out_tmpl = str(dest_dir / f"{session}.%(ext)s")
    ffmpeg = _ffmpeg_available()
    _progress(on_progress, "Downloading video (yt-dlp)…" + ("" if ffmpeg else " [no ffmpeg: full file]"))
    attempts: list[list[str]] = []

    def _add(cmd: list[str]) -> None:
        if cmd not in attempts:  # without ffmpeg several variants are identical
            attempts.append(cmd)

    # Try with captions first, then video-only (YouTube often 429s auto-subs).
    for with_subs in (True, False):
        if ffmpeg:
            _add(
                _yt_dlp_base(out_tmpl, ffmpeg=True, subs=with_subs)
                + [
                    "--download-sections",
                    f"*0-{int(max_minutes * 60)}",
                    "--force-keyframes-at-cuts",
                    "--",
                    link,
                ]
            )
        _add(_yt_dlp_base(out_tmpl, ffmpeg=ffmpeg, subs=with_subs) + ["--", link])
        _add(_yt_dlp_base(out_tmpl, ffmpeg=False, subs=with_subs) + ["--", link])
    last_err = ""
    for i, cmd in enumerate(attempts):
        # First attempt gets the long budget; retries are for auth/caption
        # problems that fail fast, so don't let them stack up to 90 minutes.
        timeout = 900 if i == 0 else 420
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",  # yt-dlp prints UTF-8 titles; cp1252 decode used to abort the run
                errors="replace",
                timeout=timeout,
            )
            last_err = ""
            break
        except FileNotFoundError as exc:
            raise RuntimeError(
                "yt-dlp not found. Run Setup GUI Agent.bat (installs yt-dlp into .venv-agent)."
            ) from exc
        except subprocess.TimeoutExpired:
            last_err = "download timed out"
        except subprocess.CalledProcessError as exc:
            last_err = (exc.stderr or exc.stdout or str(exc)).strip().splitlines()[-1:] or [str(exc)]
            last_err = str(last_err[0])[:300]
        if i + 1 < len(attempts):
            hint = last_err[:80]
            if _subtitle_fetch_failed(last_err):
                hint = "captions blocked; retrying video-only"
            _progress(on_progress, f"Retrying download ({hint})…")
    if last_err:
        demo = _last_local_demo_video(root)
        if demo is not None:
            _progress(
                on_progress,
                "YouTube unavailable — learning from your last Recorded demo instead…",
            )
            return demo, [], f"local demo ({demo.stem})"
        raise RuntimeError(_friendly_ytdlp_error(last_err))

    videos = sorted(dest_dir.glob(f"{session}.*"), key=lambda p: p.stat().st_mtime, reverse=True)
    video_path = next(
        (p for p in videos if p.suffix.lower() in {".mp4", ".webm", ".mkv", ".avi"}),
        None,
    )
    if video_path is None:
        demo = _last_local_demo_video(root)
        if demo is not None:
            _progress(
                on_progress,
                "Download produced no file — learning from your last Recorded demo instead…",
            )
            return demo, [], f"local demo ({demo.stem})"
        raise RuntimeError("Download finished but no video file was found")

    cues = []
    for p in dest_dir.glob(f"{session}*"):
        if p.suffix.lower() in {".vtt", ".srt"}:
            cues = _read_caption_file(p)
            if cues:
                break
    title = video_path.stem
    return video_path, cues, title


def _concept_budget(max_minutes: float) -> int:
    """Longer tutorials → more distinct techniques (capped so a 20h lesson stays tractable)."""
    minutes = float(max(1.0, max_minutes))
    # ~0.8 concepts per minute, min 24, max 600 (multi-hour / day-long lessons).
    return int(max(24, min(600, round(minutes * 0.8))))


def _segment_length_sec(max_minutes: float) -> float:
    """Chunk long videos so each VLM pass sees a focused window."""
    minutes = float(max(1.0, max_minutes))
    if minutes <= 12:
        return minutes * 60.0
    if minutes <= 60:
        return 10.0 * 60.0
    if minutes <= 180:
        return 12.0 * 60.0
    if minutes <= 300:
        return 15.0 * 60.0
    return 20.0 * 60.0


def concepts_from_captions(
    cues: list[tuple[float, str]],
    *,
    t0: float,
    t1: float,
) -> list[dict[str, Any]]:
    """Seed concepts from narration keywords when the vision model under-reports."""
    text = captions_window(cues, t0, t1, limit=200)
    if not text.strip():
        return []
    found: list[dict[str, Any]] = []
    seen: set[str] = set()
    for pattern, name, summary in _CAPTION_TECHNIQUES:
        if name.lower() in seen:
            continue
        if not pattern.search(text):
            continue
        seen.add(name.lower())
        found.append(
            {
                "name": name,
                "summary": summary,
                "when_to_use": f"When the lesson demonstrates {name}",
                "preconditions": ["mesh selected"],
                "aliases": [],
                "params": {},
                "t_start": float(t0),
                "t_end": float(t1),
                "steps": [],
            }
        )
    return found


def sample_keyframes(
    video_path: Path,
    *,
    max_minutes: float = 6.0,
    target_fps: float = 0.5,
    max_frames: int | None = None,
) -> list[tuple[float, Any]]:
    """Return [(t_rel, bgr_frame)] sparsely sampled, with simple cut detection."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 24.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = (total / fps) if total > 0 else float(max_minutes * 60)
        duration = min(duration, float(max_minutes) * 60.0)
        # Scale sample count with length so multi-hour lessons still get coverage.
        if max_frames is None:
            max_frames = int(max(24, min(900, 24 + float(max_minutes) * 0.7)))
        interval = max(1.0 / float(target_fps), duration / max(float(max_frames), 1.0), 1.5)
        frames: list[tuple[float, Any]] = []
        prev_gray = None
        # Probe at a third of the sample interval so a scene cut between two
        # regular samples is caught; take a frame on a cut OR when the regular
        # interval has elapsed since the last kept frame.
        probe = max(0.5, interval / 3.0)
        t = 0.0
        while t <= duration and len(frames) < max_frames:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                t += probe
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
            if prev_gray is None:
                take = True
            else:
                diff = float(np.mean(cv2.absdiff(prev_gray, gray))) / 255.0
                since_last = t - frames[-1][0] if frames else interval
                take = diff >= 0.04 or since_last >= interval * 0.95
            if take:
                frames.append((t, downsample_for_vlm(frame, max_side=MAX_KEYFRAME_SIDE)))
            prev_gray = gray
            t += probe
        return frames
    finally:
        cap.release()


def extract_concept_candidates(
    *,
    model: str,
    goal: str,
    captions: str,
    frames: list[tuple[float, Any]],
    on_progress: ProgressFn | None = None,
) -> list[dict[str, Any]]:
    _progress(on_progress, "Finding concepts in the lesson…")
    images: list[str] = []
    for _t, frame in frames[:: max(1, len(frames) // 4)][:4]:
        small = downsample_for_vlm(frame, max_side=512)
        images.append(encode_jpeg_b64(small, quality=70))
    user = (
        f"Student goal (optional focus): {goal or '(general modeling techniques)'}\n\n"
        f"Captions / narration excerpt:\n{captions[:3500]}\n\n"
        "List EVERY distinct Blender modeling technique taught in this segment "
        "(not only extrude/inset/bevel — include modifiers, knife, bridge, boolean, join, etc.)."
    )
    raw = ""
    try:
        raw = chat_vision_json(
            model,
            system=CONCEPT_LIST_SYSTEM,
            user_text=user,
            images_b64=images,
            num_predict=2400,
        )
    except PolicyError as exc:
        # Text-only model or vision failure: the narration alone is still a good teacher.
        _progress(on_progress, f"Vision failed ({str(exc)[:60]}); learning from narration…")
        raw = chat_text_json(model, system=CONCEPT_LIST_SYSTEM, user_text=user, num_predict=2400)
    data = _parse_json_object(raw)
    concepts = data.get("concepts") if isinstance(data, dict) else None
    if not isinstance(concepts, list):
        return []
    return merge_concepts([c for c in concepts if isinstance(c, dict)])


def fill_concept_steps(
    concept: dict[str, Any],
    *,
    model: str,
    frames: list[tuple[float, Any]],
    cues: list[tuple[float, str]],
    on_progress: ProgressFn | None = None,
) -> dict[str, Any]:
    name = str(concept.get("name") or "concept")
    _progress(on_progress, f"Teaching steps for '{name}'…")
    t0 = float(concept.get("t_start") or 0.0)
    t1 = float(concept.get("t_end") or (t0 + 30.0))
    if t1 <= t0:
        t1 = t0 + 30.0
    window = [f for t, f in frames if t0 - 2.0 <= t <= t1 + 2.0]
    if not window:
        window = [f for _, f in frames[:2]]
    images = [encode_jpeg_b64(downsample_for_vlm(f, max_side=512), quality=70) for f in window[:3]]
    cap = captions_window(cues, t0, t1)
    user = (
        f"Concept name: {name}\n"
        f"Summary: {concept.get('summary') or ''}\n"
        f"When to use: {concept.get('when_to_use') or ''}\n"
        f"Preconditions: {concept.get('preconditions') or []}\n"
        f"Narration around this concept:\n{cap}\n\n"
        "Produce canonical Hands steps for applying this on the USER's current mesh."
    )
    steps: list[dict[str, Any]] = []
    for attempt in ("vision", "text"):
        try:
            if attempt == "vision":
                raw = chat_vision_json(
                    model, system=CONCEPT_STEPS_SYSTEM, user_text=user, images_b64=images, num_predict=400,
                )
            else:
                raw = chat_text_json(model, system=CONCEPT_STEPS_SYSTEM, user_text=user, num_predict=400)
            data = _parse_json_object(raw)
            steps_raw = data.get("steps") if isinstance(data, dict) else None
            steps = sanitize_concept_steps(list(steps_raw) if isinstance(steps_raw, list) else [])
            if steps:
                break
        except PolicyError:
            steps = []
    if not steps:
        # Safe fallback: F3 search for the concept name.
        steps = sanitize_concept_steps(
            [
                {"action": "key", "keys": ["f3"], "reason": f"search {name}"},
                {"action": "type", "text": name.title(), "reason": f"find {name}"},
                {"action": "key", "keys": ["enter"], "reason": f"run {name}"},
            ]
        )
    out = dict(concept)
    out["steps"] = steps
    return ground_concept(out)


def learn_concepts_from_video(
    *,
    url: str = "",
    path: str = "",
    goal: str = "",
    model: str = "llava",
    max_minutes: float = 6.0,
    dataset_root: Path | None = None,
    on_progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """Full pipeline: fetch → keyframes → multi-segment concept cards with steps."""
    started = time.time()
    max_minutes = float(max(1.0, min(1200.0, max_minutes)))
    video_path, cues, title = fetch_video_source(
        url=url,
        path=path,
        dataset_root=dataset_root,
        max_minutes=max_minutes,
        on_progress=on_progress,
    )
    _progress(on_progress, "Sampling keyframes…")
    frames = sample_keyframes(video_path, max_minutes=max_minutes)
    if not frames:
        raise RuntimeError("No frames could be read from the video")

    duration = min(float(frames[-1][0]) + 1.0, float(max_minutes) * 60.0)
    seg_len = _segment_length_sec(max_minutes)
    segments: list[tuple[float, float]] = []
    t = 0.0
    while t < duration - 0.5:
        segments.append((t, min(t + seg_len, duration)))
        t += seg_len
    if not segments:
        segments = [(0.0, duration)]

    all_candidates: list[dict[str, Any]] = []
    for si, (t0, t1) in enumerate(segments):
        _progress(
            on_progress,
            f"Scanning segment {si + 1}/{len(segments)} ({t0 / 60:.0f}–{t1 / 60:.0f} min)…",
        )
        seg_frames = [(tt, f) for tt, f in frames if t0 - 0.5 <= tt <= t1 + 0.5]
        if not seg_frames:
            mid = 0.5 * (t0 + t1)
            seg_frames = sorted(frames, key=lambda row: abs(row[0] - mid))[:4]
        cap_text = captions_window(cues, t0, t1, limit=160)
        if not cap_text:
            focus = (goal or "").strip() or title
            cap_text = (
                f"(no captions) Video title: {title}. Segment {si + 1}/{len(segments)}. "
                f"Goal focus: {focus}. List EVERY distinct Blender technique shown "
                f"(boolean, knife, bridge, solidify, array, mirror, extrude, inset, bevel, "
                f"loop cut, join, separate, modifiers…)."
            )
        cands = extract_concept_candidates(
            model=model,
            goal=goal,
            captions=cap_text,
            frames=seg_frames,
            on_progress=on_progress,
        )
        for c in cands:
            c = dict(c)
            if float(c.get("t_start") or 0) <= 0 and float(c.get("t_end") or 0) <= 0:
                c["t_start"] = t0
                c["t_end"] = t1
            all_candidates.append(c)
        for c in concepts_from_captions(cues, t0=t0, t1=t1):
            all_candidates.append(c)

    candidates = merge_concepts(all_candidates)
    if not candidates:
        # Never seed a concept with the user's Run goal ("make me a house") —
        # that becomes an F3-search stub and poisons later planning.
        from agent.planner import is_complex_object_goal

        seed = (title or "modeling technique").strip()
        if is_complex_object_goal(seed) or is_complex_object_goal(goal or ""):
            seed = "mesh modeling technique"
        candidates = [
            {
                "name": seed[:48],
                "summary": f"Technique inferred from {title}",
                "when_to_use": "When refining mesh detail",
                "preconditions": ["mesh selected"],
                "aliases": [],
                "t_start": 0.0,
                "t_end": duration,
                "steps": [],
            }
        ]

    budget = _concept_budget(max_minutes)
    filled: list[dict[str, Any]] = []
    take = candidates[:budget]
    # LLM step-teaching is the slow part; ground the rest from the knowledge base
    # so a couple-hundred concept harvest does not mean hundreds of VLM calls.
    step_llm_cap = min(len(take), 96)
    for i, card in enumerate(take):
        card = dict(card)
        card["video"] = str(video_path.name)
        if i < step_llm_cap:
            _progress(on_progress, f"Teaching {i + 1}/{len(take)} · {card.get('name')}")
            filled.append(
                fill_concept_steps(
                    card,
                    model=model,
                    frames=frames,
                    cues=cues,
                    on_progress=on_progress,
                )
            )
        else:
            _progress(on_progress, f"Grounding {i + 1}/{len(take)} · {card.get('name')}")
            if not card.get("steps"):
                name = str(card.get("name") or "technique")
                card["steps"] = sanitize_concept_steps(
                    [
                        {"action": "key", "keys": ["f3"], "reason": f"search {name}"},
                        {"action": "type", "text": name.title(), "reason": f"find {name}"},
                        {"action": "key", "keys": ["enter"], "reason": f"run {name}"},
                    ]
                )
            filled.append(ground_concept(card))

    merged = [c if c.get("ops") else ground_concept(c) for c in merge_concepts(filled)]
    # Only report concepts that memory will actually keep (search-only stubs are dropped).
    from agent.planner import is_search_only_ops

    merged = [c for c in merged if c.get("ops") and not is_search_only_ops(list(c.get("ops") or []))]
    names = ", ".join(c["name"] for c in merged[:24])
    _progress(on_progress, f"Learned {len(merged)} concept(s): {names}")
    # Downloaded tutorials (up to 1200 min of 720p) are not needed after sampling.
    _prune_teacher_videos(video_path, dataset_root)
    return {
        "ok": True,
        "video": str(video_path),
        "title": title,
        "concepts": merged,
        "elapsed_sec": time.time() - started,
        "segments": len(segments),
    }