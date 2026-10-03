# Local Text to 3D

A **Blender 5.1 addon** that turns a text prompt into a mesh, entirely on the computer where you install it. No Meshy, no Tripo, no API keys, no paid service. An optional **local GUI Agent** can drive Blender with an Ollama vision model (mouse and keyboard).

The addon is a sidebar panel. Mesh generation runs in a **local worker** so Blender's Python never loads PyTorch. The GUI Agent runs as a separate localhost sidecar.

Each person clones this repository and runs the worker and sidecar **on their own machine**. The addon on that machine talks only to:

- Worker: `http://127.0.0.1:8765`
- GUI Agent sidecar: `http://127.0.0.1:8766`

Those ports are **local defaults**. They are not an address for someone else's computer.

## Share safely

Share the code or a release archive. Do not share remote access to a running install.

- Do **not** connect the addon to someone else's computer.
- Do **not** port-forward `8765` or `8766`.
- Do **not** publish or send your public IP, LAN IP, or Tailscale IP. None of those belong in setup.
- Bind the worker and sidecar to `127.0.0.1` only. The programs refuse `0.0.0.0` and other non-loopback addresses.
- In Blender, leave **Worker URL** at `http://127.0.0.1:8765` and **Sidecar URL** at `http://127.0.0.1:8766`. The addon accepts only `127.0.0.1`, `localhost`, or `::1`, and only on those two ports. A LAN or WAN host is rejected.
- Paths in preferences, `.env`, and the agent dataset stay on your machine. Use placeholders such as `/path/to/LocalText3D` if you write docs or examples. Do not commit a filled-in `.env` (it is gitignored).

## Install on your computer

You need Blender 5.1 and **64-bit Python 3.13** from [python.org](https://www.python.org/downloads/) (tick **Add python.exe to PATH** on Windows). A current NVIDIA driver is needed for GPU generation. CPU and mock modes work without one.

Blender already has its own Python. The Python you install is only for the **worker** (PyTorch / Shap-E) and the optional agent sidecar.

Clone or unpack the repo anywhere you like. Examples: `C:\path\to\LocalText3D` or `/path/to/LocalText3D`. Every command below is run from that folder.

On a **16 GB-class NVIDIA card**, start with **Shap-E** (`.venv`). Install **TRELLIS** into `.venv-trellis` (Python 3.10 + Windows wheels) when you want higher quality. Both stay local and free. The worker keeps only one of them in VRAM at a time.

### Windows

```powershell
py -3.13 scripts\run_all.py
```

Or double-click `START.bat` or `scripts\run_all.bat`.

That creates `.venv` with CUDA 12.6 PyTorch + Shap-E (Python 3.13), writes `dist\localtext3d-<version>.zip` (the version comes from `addon\blender_manifest.toml`; the script prints the exact file), and serves `http://127.0.0.1:8765` on **this** computer. First run downloads several GB. Leave the window open.

### Linux

```bash
cd /path/to/LocalText3D
python3 scripts/run_all.py
```

Or `./start.sh`. Same result: a `.venv`, an addon zip under `dist/`, and a worker on `http://127.0.0.1:8765`.

CPU-only torch:

```bash
python3 scripts/setup_worker.py --cpu
.venv/bin/python -m worker serve
python3 scripts/build_addon.py
```

`python3 scripts/setup_worker.py` without `--cpu` uses CUDA when an NVIDIA GPU is available.

### Mock mode (no model download)

```bash
python3 -m worker serve --mock
python3 -m worker generate "a mug" --engine shap_e --mock --output mug.glb
```

From the repo folder, with the worker package on `PYTHONPATH` (or via `scripts/run_all.py --mock`). Every engine writes a placeholder cube. The panel still switches engines. The server still binds to `127.0.0.1:8765`.

### Install the addon

In Blender 5.1: **Edit → Preferences → Get Extensions → Install from Disk** → the `localtext3d-<version>.zip` in `dist/`. Enable **Local Text to 3D**.

Turn **Online Access** on. Blender needs that permission even for localhost.

Addon preferences:

| Setting | Value on your machine |
| --- | --- |
| Worker URL | `http://127.0.0.1:8765` |
| LocalText3D project folder | the clone, for example `/path/to/LocalText3D` (only if you start the sidecar from Blender) |
| Megascans / Fab folder | optional; a folder of files you already downloaded |

### Generate

1. Press `N` in the 3D Viewport and open the **Text to 3D** tab.
2. Click **Check Worker**. It should list `shap_e` (and `trellis` after the TRELLIS setup).
3. Set **Source** to **Text**, **Image**, **Viewport**, **Orbit**, or **Scan**. Set **Engine** to **Shap-E** or **TRELLIS**.
4. Prompt example: `a wooden stool with three legs`. For Viewport/Orbit, look at a blockout in the 3D view instead. **Text** auto-picks a matching local Megascans / Fab / Quixel Bridge asset when one exists (already-downloaded files only). **Scan → Selected** variants an imported mesh.
5. Click **Generate 3D Model**. Image/Viewport/Orbit uses `TRELLIS-image-large`. The GLB is imported at 1 meter, sat on the ground at the 3D cursor, and lined up beside the last generate so seeds do not stack. Re-import or restore a prompt from **Keepers** even after the worker restarts.

Until TRELLIS is installed, a TRELLIS generate returns a clear "engine not installed" error. Shap-E still works.

Do **not** GPU-render in Cycles while a job is running.

## What the panel does

```
Blender 5.1 addon  -->  http://127.0.0.1:8765  -->  one local model in VRAM
     N-panel            (this computer only)       TRELLIS  or  Shap-E
     import GLB                                      never both at once
```

| Engine | Role | Typical VRAM | First download | Install |
| --- | --- | --- | --- | --- |
| Shap-E | Faster draft | ~4–8 GB | ~3.5 GB | `.venv` (Python 3.13) |
| TRELLIS `text-large` | Better mesh and texture | ~12–16 GB fp16 | Several GB | `.venv-trellis` (Python 3.10, Windows) |
| TRELLIS `text-base` | Fallback if large OOMs | ~8 GB | Smaller | `.venv-trellis` |

After each engine's weights are cached, the worker can run offline (`HF_HUB_OFFLINE=1`).

## TRELLIS on Windows (native)

No WSL. One-time install (several GB):

```powershell
py -3.13 scripts\setup_trellis.py
```

That creates `.venv-trellis` (Python 3.10), clones `vendor/TRELLIS`, installs CUDA wheels, and adds Shap-E to the same env. Then:

```powershell
py -3.13 scripts\run_all.py --skip-setup
```

or double-click `START.bat`. `GET /health` on `http://127.0.0.1:8765` should list `shap_e` and `trellis`. First TRELLIS generate downloads `microsoft/TRELLIS-text-large`. Do not default to `text-xlarge` on 16 GB.

Equivalent one-shot:

```powershell
py -3.13 scripts\run_all.py --with-trellis
```

`--with-trellis` is the Windows installer. It is not a remote setup.

## Offline use

1. Run one successful generate per engine you care about.
2. Start the worker with `HF_HUB_OFFLINE=1`.
3. Keep the Worker URL at `http://127.0.0.1:8765`.

Outputs land in `%LOCALAPPDATA%\localtext3d\outputs` on Windows and `~/.cache/localtext3d/outputs` on Linux. Override with `LOCALTEXT3D_OUTPUT_DIR` set to a folder on your machine, for example `/path/to/localtext3d-outputs`. See `.env.example`. The addon only accepts a GLB that resolves inside that output directory.

## Worker API

Listens on `127.0.0.1:8765` only. Requests whose `Host` or `Origin` is not loopback are rejected. `--host 0.0.0.0` and LAN addresses exit with an error.

- `GET /health` — device, VRAM, available engines, loaded engine
- `GET /jobs` — recent finished meshes (prompt, seed, GLB path). Survives a worker restart
- `POST /generate` — `{engine, prompt, seed, ...}` → `{job_id}`
- `GET /jobs/{id}` — status and output path
- `GET /jobs/{id}/file` — download the GLB

Only one job runs at a time. The Blender addon will not follow this API to any host other than loopback, and not to any port other than `8765`.

## GUI Agent (local vision control)

Optional sidecar that watches the Blender window and acts with mouse and keyboard via a local Ollama vision model. You can **Record** demos, **Learn from Video** (YouTube or a local file), then **Run** a goal. The agent composes a Blender plan (primitives and techniques) instead of only replaying one demo.

Ollama is separate and also local: `http://127.0.0.1:11434`. That is not the worker and not the sidecar.

### Setup (once), on your machine

1. Install [Ollama](https://ollama.com) and pull a vision model:

```powershell
ollama pull llama3.2-vision
```

2. Create the agent venv (first time only), from `/path/to/LocalText3D`:

```powershell
py -3.13 scripts\setup_agent.py
```

On Linux: `python3 scripts/setup_agent.py`. Or double-click `Setup GUI Agent.bat`.

3. Start the sidecar and leave it running:

```powershell
.\.venv-agent\Scripts\python.exe -m agent serve
```

On Linux: `.venv-agent/bin/python -m agent serve`. Or double-click `Start GUI Agent.bat`.

It serves `http://127.0.0.1:8766` on this computer. The addon accepts only that port.

To start the sidecar from Blender, set **LocalText3D project folder** to your clone (`/path/to/LocalText3D` or `C:\path\to\LocalText3D`). That preference is stored by Blender on your machine. It is not in this repository.

### Use in Blender

1. Enable **Local Text to 3D** and turn **Online Access** on.
2. Press `N` → **Text to 3D** → **GUI Agent**.
3. Click **Check GUI Agent** (should prefer `llama3.2-vision` when installed). Use **Start Sidecar** if it is not running yet.
4. Enter a goal (for example `make me a house`, `extrude the selected face`, or `add a cone`).
5. Pick the vision / planner models → **Run Agent**.

Keep Blender visible. The agent focuses the Blender window and runs until the plan finishes, you hit Stop / Esc, or the step budget is exhausted.

### Learn from video

- Paste a **YouTube URL** or pick a local video, set **Minutes** (up to **1200 = 20 hours**), then **Learn from Video**.
- A local file is copied into the agent dataset before the sidecar opens it. The sidecar refuses videos outside that folder.
- Default dataset: `~/LocalText3D/agent_dataset` (your home directory). Override with `LOCALTEXT3D_AGENT_DATASET`, for example `/path/to/agent_dataset`.
- Concepts are stored as Blender **techniques** (bevel, extrude, loop cut, and so on) and grafted onto later object plans when relevant. They are not whole-object clones.

### Modes

| Mode | When |
| --- | --- |
| Compose / skills | Object goals get a silhouette plan; learned techniques graft on when they match |
| Skill / memory | Goal matches a recorded skill or prior successful plan |
| Recipe | Tiny built-in hotkey scripts (for example Add Cube via F3) |
| Freeform vision | No matching skill — `llama3.2-vision` decides each step from the screen |
| Replay | Explicitly replay the last recording |

### Planning notes (1.6.4)

- Plan length and **Max steps** scale with how complex the goal is (simple phrases stay short; bridges and detailed houses get hundreds of GUI actions).
- Technique matching ignores hotkey junk and object-name collisions (for example "make a bridge" does not grab "Bridge edge loops" just because of the word *bridge*).
- The Plan panel lists many ops (not a fixed 40-line cut-off of the real plan).

Recording demos is optional. It still helps for hard skills, but you can run freeform goals without teaching first.

### Agent API

Bound to `127.0.0.1:8766` only. Same loopback rule as the worker: non-loopback binds are refused.

- `GET /health` — Ollama reachability, models, preferred vision model, agent status
- `GET /status` — running / recording / step / goal / mode

### Agent tests

From the repo, after `scripts/setup_agent.py`:

```powershell
.\.venv-agent\Scripts\python.exe -m unittest tests.test_agent_core tests.test_agent_semantic
```

On Linux: `.venv-agent/bin/python -m unittest tests.test_agent_core tests.test_agent_semantic`.

## Tests

Worker tests do not need a GPU:

```bash
python3 -m unittest tests.test_worker
```

## Local configuration

`.env.example` lists optional environment variables with placeholder paths. Copy it to `.env` only on your machine. `.env` is gitignored, as are the agent dataset files (`episodes.jsonl`, `memory.jsonl`, `last_session.json`) and Blender `.blend` files, which can contain absolute paths.

## License

The addon is **GPL-3.0-or-later**. TRELLIS and Shap-E weights are MIT and are downloaded by the worker, not shipped in this repo.
