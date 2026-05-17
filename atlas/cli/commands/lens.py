"""atlas lens — Geometric Lens probe + build pipeline (PC-057, PC-058).

Two subcommands wrap the existing geometric-lens training code into a
model-path-driven workflow so users can bring their own GGUF and either
verify it's Lens-compatible (`check`) or actually train fresh artifacts
for it (`build`).

Layering:
    PC-057 `atlas lens check`  -> this file, cheap pre-flight
    PC-058 `atlas lens build`  -> this file, wraps training.train_cost_field
    PC-059 `atlas lens push`   -> roadmap, publishes to registry
    PC-060 HF middleman        -> roadmap, automated distribution

Probe contract: both subcommands talk to a *running* llama-server via
its `/embedding` and `/props` endpoints. ATLAS users typically already
have one up (`docker compose up -d`); if not, the commands print a
clear "start the stack first" hint rather than spinning their own
process. Keeping this stateless against an existing server matches the
rest of the atlas CLI surface (model.py, doctor.py, tier.py all assume
some level of running infrastructure for their richer probes).

Invoke:
    atlas lens check                    # probe currently-loaded model
    atlas lens check Qwen3.5-9B-Q6_K    # probe a registry entry
    atlas lens check /path/to/model.gguf  # probe an arbitrary file
    atlas lens build <name|path>        # train fresh C(x) artifacts
    atlas lens --json                   # machine-readable output for scripts

Exit codes (check):
    0  compat        — artifacts exist + dim matches + server reachable
    1  needs-build   — model loadable, no artifacts at right dim
    2  incompatible  — can't probe (server down, model won't load, PC-202 missing)
"""

import argparse
import json as jsonlib
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

from atlas.cli.commands import model_registry


# Reuse the color + safe-print primitives from tier/doctor for consistency.
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELL = "\033[33m"
CYAN = "\033[36m"


def _supports_unicode() -> bool:
    enc = (getattr(sys.stdout, "encoding", None) or "").lower()
    return "utf" in enc


UNICODE_OK = _supports_unicode()


def _safe_print(s: str = "") -> None:
    try:
        print(s)
    except UnicodeEncodeError:
        print(s.encode("ascii", errors="replace").decode("ascii"))


# ---------------------------------------------------------------------------
# llama-server probe helpers
# ---------------------------------------------------------------------------

# C(x) constructor permits arbitrary input_dim; what's "compat" is whether
# the saved artifact's first-layer weight matches the model's embedding dim.
# Keeping a canonical value documents the V3.1.0 baseline.
LENS_CANONICAL_DIM = 4096  # Qwen3.5-9B hidden size


def _llama_url() -> str:
    """Resolve where llama-server is listening.

    Mirrors geometric-lens/embedding_extractor.py's resolution order so
    `atlas lens check` agrees with what the lens service itself sees.
    """
    return os.environ.get(
        "ATLAS_LLAMA_URL",
        os.environ.get(
            "LLAMA_EMBED_URL",
            os.environ.get("LLAMA_URL", "http://localhost:8080"),
        ),
    )


@dataclass
class LlamaProbe:
    """Snapshot of what the running llama-server can tell us about the model."""
    reachable: bool
    url: str
    embedding_dim: int = 0          # 0 when /embedding failed or didn't return
    n_layers: int = 0               # 0 when /props didn't carry n_layer
    model_name: str = ""            # whatever /props reports (often a path)
    has_hidden_states_patch: bool = False  # PC-202: layers extension present
    error: str = ""                 # short human description when reachable=False


def _http_get(url: str, timeout: float = 5.0) -> Optional[dict]:
    """GET a JSON endpoint. Returns parsed dict or None on any failure."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return jsonlib.loads(resp.read())
    except (urllib.error.URLError, urllib.error.HTTPError,
            jsonlib.JSONDecodeError, OSError, ValueError):
        return None


def _http_post_json(url: str, body: dict, timeout: float = 30.0) -> Optional[dict]:
    """POST JSON, parse response. Returns parsed obj or None on failure."""
    payload = jsonlib.dumps(body).encode()
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return jsonlib.loads(resp.read())
    except (urllib.error.URLError, urllib.error.HTTPError,
            jsonlib.JSONDecodeError, OSError, ValueError):
        return None


def probe_llama(url: Optional[str] = None,
                sample_text: str = "def hello():\n    return 42") -> LlamaProbe:
    """Discover what the running llama-server knows about its loaded model.

    Three probes:
      1. /health  -> is the server reachable at all?
      2. /props   -> model metadata (n_layer, model_name)
      3. /embedding (POST) -> the authoritative embedding dim, plus a
         PC-202 hidden-states ping (layers=[0]) to detect the patch.

    Failures degrade gracefully: a probe step that times out or returns
    a non-JSON body sets that field to its zero value rather than
    raising. Caller inspects reachable + embedding_dim to decide verdict.
    """
    url = url or _llama_url()
    probe = LlamaProbe(reachable=False, url=url)

    # 1. /health — fast existence check
    health = _http_get(f"{url}/health", timeout=3.0)
    if health is None:
        probe.error = (f"llama-server not reachable at {url}. "
                       f"Bring the stack up with `docker compose up -d` "
                       f"(or `docker compose -f docker-compose.yml "
                       f"-f docker-compose.rocm.yml up -d` on AMD), "
                       f"then re-run.")
        return probe
    probe.reachable = True

    # 2. /props — n_layer + model name. Field names changed across
    # llama-server versions; tolerate both `n_layer` and `default_generation_settings`.
    props = _http_get(f"{url}/props", timeout=5.0) or {}
    probe.n_layers = (
        int(props.get("n_layer", 0))
        or int((props.get("default_generation_settings") or {}).get("n_layer", 0))
    )
    probe.model_name = (props.get("model_path")
                        or props.get("model_name")
                        or props.get("model")
                        or "")

    # 3. /embedding — authoritative dim. Send a small sample; pooled or
    # per-token both yield the dim. The PC-202 patch is signalled by the
    # presence of `hidden_states_dim` in the response when `layers` is
    # requested; on an unpatched server the field is silently absent.
    emb = _http_post_json(f"{url}/embedding",
                          {"content": sample_text, "layers": [0]},
                          timeout=30.0)
    if isinstance(emb, list) and emb:
        first = emb[0]
        if isinstance(first, dict):
            raw = first.get("embedding")
            if isinstance(raw, list) and raw:
                if isinstance(raw[0], list):
                    probe.embedding_dim = len(raw[0])  # per-token
                else:
                    probe.embedding_dim = len(raw)     # pooled
            if "hidden_states_dim" in first:
                probe.has_hidden_states_patch = True
    if probe.embedding_dim == 0:
        probe.error = (f"llama-server at {url} is up but /embedding "
                       f"didn't return an embedding. Likely cause: model "
                       f"was started without `--embeddings`. Check "
                       f"inference/entrypoint-v3.1-9b.sh.")
    return probe


# ---------------------------------------------------------------------------
# Artifact resolution + dim inspection
# ---------------------------------------------------------------------------

def _resolve_model_arg(arg: Optional[str]) -> Optional[model_registry.Model]:
    """Best-effort lookup: registry name → Model, or path/None → None.

    `atlas lens check` accepts:
      - a registry name        (e.g. "Qwen3.5-9B-Q6_K")
      - a .gguf path           (any model on disk)
      - nothing                (probe whatever llama-server has loaded)
    """
    if not arg:
        return None
    for m in model_registry.REGISTRY:
        if m.name == arg or m.model_file == os.path.basename(arg):
            return m
    return None


@dataclass
class ArtifactInspection:
    """Result of looking at the on-disk Lens artifact."""
    present: bool                      # cost_field.pt exists on disk
    dim: Optional[int] = None          # input dim if introspectable
    torch_available: bool = True       # False -> dim couldn't be checked
    error: str = ""


def _inspect_cost_field(artifact_dir: str) -> ArtifactInspection:
    """Look at cost_field.pt and report what we can.

    Three outcomes:
      1. File missing                -> present=False
      2. File present, torch missing -> present=True, dim=None,
                                         torch_available=False
      3. File present, torch present -> present=True, dim=<int>
                                         (or None on a load error,
                                          with `error` set)
    Distinguishing (2) from (3) lets the verdict avoid misleading users
    into a needs-build state when the artifact really exists but the
    host Python just can't peek at it.
    """
    cost_path = os.path.join(artifact_dir, "cost_field.pt")
    if not os.path.isfile(cost_path):
        return ArtifactInspection(present=False)
    try:
        import torch
    except ImportError:
        return ArtifactInspection(present=True, dim=None,
                                  torch_available=False,
                                  error="torch not installed on host")
    try:
        state = torch.load(cost_path, map_location="cpu", weights_only=True)
    except Exception:
        try:
            state = torch.load(cost_path, map_location="cpu")
        except Exception as e:
            return ArtifactInspection(present=True, dim=None,
                                      error=f"torch.load failed: {e}")
    if not isinstance(state, dict):
        return ArtifactInspection(present=True, dim=None,
                                  error="state dict has unexpected shape")
    # CostField.net.0 is the first Linear layer; its weight is (out, in).
    for key in ("net.0.weight", "0.weight"):
        if key in state:
            try:
                return ArtifactInspection(present=True,
                                          dim=int(state[key].shape[1]))
            except Exception:
                continue
    return ArtifactInspection(present=True, dim=None,
                              error="no recognized first-layer weight key")


# Back-compat shim — older callers (and the test suite) read just the dim.
def _read_saved_cost_field_dim(artifact_dir: str) -> Optional[int]:
    return _inspect_cost_field(artifact_dir).dim


# ---------------------------------------------------------------------------
# atlas lens check  (PC-057)
# ---------------------------------------------------------------------------

@dataclass
class CheckVerdict:
    verdict: str          # 'compat' | 'needs-build' | 'incompatible'
    reason: str
    probe: LlamaProbe
    artifact_dir: Optional[str] = None
    artifact_dim: Optional[int] = None
    matched_model: Optional[str] = None
    # True when the artifact is present on disk but its input dim couldn't
    # be introspected (typically because torch isn't installed on the host
    # Python). Verdict stays "compat" (don't push users to needs-build for
    # a host-tooling gap) but JSON consumers can branch on this.
    unverified: bool = False

    @property
    def exit_code(self) -> int:
        return {"compat": 0, "needs-build": 1, "incompatible": 2}.get(self.verdict, 2)


def _atlas_root() -> str:
    """Best-effort: walk up from CWD looking for docker-compose.yml, fall back to CWD."""
    cur = os.path.abspath(os.getcwd())
    while True:
        if os.path.isfile(os.path.join(cur, "docker-compose.yml")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return os.path.abspath(os.getcwd())
        cur = parent


def _check_model(arg: Optional[str], atlas_root: str) -> CheckVerdict:
    """The actual probe + verdict logic. Pure function for testability."""
    probe = probe_llama()
    if not probe.reachable:
        return CheckVerdict(verdict="incompatible", reason=probe.error,
                            probe=probe)
    if probe.embedding_dim == 0:
        return CheckVerdict(verdict="incompatible", reason=probe.error,
                            probe=probe)

    matched = _resolve_model_arg(arg)
    matched_name = matched.name if matched else None

    # Resolve artifact dir. For known-supported registry entries this is
    # already wired; for arbitrary models we fall back to ATLAS_LENS_MODELS
    # or the global default. Either way, "is there a cost_field.pt whose
    # input dim matches the model's embedding dim?" is the decisive question.
    if matched and matched.lens_status == "supported":
        artifact_dir = model_registry.lens_artifact_dir_for(matched, atlas_root)
    else:
        env = os.environ.get("ATLAS_LENS_MODELS")
        if env:
            artifact_dir = env if os.path.isabs(env) else \
                os.path.normpath(os.path.join(atlas_root, env))
        else:
            artifact_dir = os.path.normpath(os.path.join(
                atlas_root, "geometric-lens", "geometric_lens", "models"))

    inspection = (_inspect_cost_field(artifact_dir) if artifact_dir
                  else ArtifactInspection(present=False))

    if not inspection.present:
        return CheckVerdict(
            verdict="needs-build",
            reason=(f"Model produces {probe.embedding_dim}-dim embeddings, but "
                    f"no cost_field.pt found in {artifact_dir}. Run "
                    f"`atlas lens build` to train fresh artifacts."),
            probe=probe, artifact_dir=artifact_dir,
            artifact_dim=None, matched_model=matched_name,
        )

    if inspection.dim is None and not inspection.torch_available:
        # Artifact exists but the host Python can't peek at its dim. Don't
        # send the user to needs-build over a tooling gap on the host —
        # the lens service in the container has its own torch and will
        # score fine. Surface the unverified state via the dedicated flag.
        return CheckVerdict(
            verdict="compat", unverified=True,
            reason=(f"cost_field.pt exists at {artifact_dir} but the host "
                    f"Python can't introspect its dim (torch not installed). "
                    f"Assuming compat — the lens service in the container "
                    f"has torch and will score normally. `pip install torch` "
                    f"on the host if you want this to verify properly."),
            probe=probe, artifact_dir=artifact_dir,
            artifact_dim=None, matched_model=matched_name,
        )

    if inspection.dim is None:
        # Torch is available but the load failed for some other reason —
        # corrupted file, unrecognized layout, etc. Treat as needs-build
        # since we can't confirm the artifact is usable.
        return CheckVerdict(
            verdict="needs-build",
            reason=(f"cost_field.pt at {artifact_dir} could not be inspected: "
                    f"{inspection.error}. Rebuild with `atlas lens build`."),
            probe=probe, artifact_dir=artifact_dir,
            artifact_dim=None, matched_model=matched_name,
        )
    artifact_dim = inspection.dim

    if artifact_dim != probe.embedding_dim:
        return CheckVerdict(
            verdict="needs-build",
            reason=(f"Dim mismatch: model emits {probe.embedding_dim}-dim "
                    f"embeddings but the saved cost_field.pt expects "
                    f"{artifact_dim}-dim input. Run `atlas lens build` to "
                    f"train fresh artifacts at the model's native dim."),
            probe=probe, artifact_dir=artifact_dir,
            artifact_dim=artifact_dim, matched_model=matched_name,
        )

    # Dim matches. PC-202 hidden-states patch is nice-to-have for G(x)
    # metric tensor work but not required for C(x) scoring; report it
    # as a warning surface rather than a hard failure.
    note = ""
    if not probe.has_hidden_states_patch:
        note = (" Note: PC-202 hidden-states patch not detected on llama-server. "
                "C(x) works fine; G(x) metric-tensor training would need a "
                "patched build (inference/Dockerfile.v31).")
    return CheckVerdict(
        verdict="compat",
        reason=(f"Model emits {probe.embedding_dim}-dim embeddings; "
                f"cost_field.pt at {artifact_dir} accepts {artifact_dim}-dim. "
                f"Ready to score.{note}"),
        probe=probe, artifact_dir=artifact_dir,
        artifact_dim=artifact_dim, matched_model=matched_name,
    )


def _emit_check(args: argparse.Namespace, color: bool) -> int:
    atlas_root = _atlas_root()
    v = _check_model(args.model, atlas_root)

    if args.json:
        out = asdict(v)
        out["probe"] = asdict(v.probe)
        out["exit_code"] = v.exit_code
        print(jsonlib.dumps(out, indent=2))
        return v.exit_code

    badge = {
        "compat":       f"{GREEN}compat{RESET}"       if color else "compat",
        "needs-build":  f"{YELL}needs-build{RESET}"   if color else "needs-build",
        "incompatible": f"{RED}incompatible{RESET}"   if color else "incompatible",
    }[v.verdict]
    hdr = f"{BOLD}atlas lens check{RESET}" if color else "atlas lens check"
    _safe_print(f"{hdr}  verdict: {badge}")
    _safe_print(f"  llama-server: {v.probe.url} "
                f"({'reachable' if v.probe.reachable else 'unreachable'})")
    if v.probe.reachable:
        _safe_print(f"  model:        {v.probe.model_name or '(unknown)'}")
        _safe_print(f"  embedding:    {v.probe.embedding_dim}-dim")
        _safe_print(f"  layers:       {v.probe.n_layers or '(unknown)'}")
        _safe_print(f"  PC-202 patch: "
                    f"{'yes' if v.probe.has_hidden_states_patch else 'no'}")
    if v.artifact_dir:
        _safe_print(f"  artifact dir: {v.artifact_dir}")
    if v.artifact_dim is not None:
        _safe_print(f"  artifact dim: {v.artifact_dim}-dim")
    if v.matched_model:
        _safe_print(f"  registry hit: {v.matched_model}")
    _safe_print("")
    _safe_print(f"  {v.reason}")
    return v.exit_code


# ---------------------------------------------------------------------------
# atlas lens build  (PC-058)
# ---------------------------------------------------------------------------

def _extract_training_embeddings(samples: List[Dict],
                                  llama_url: str,
                                  color: bool) -> Dict:
    """For each sample {text, label}, POST /embedding and collect.

    Returns a dict compatible with training.train_cost_field's `data` arg:
        {"embeddings": [List[float], ...], "labels": [0|1, ...]}
    """
    embeddings: List[List[float]] = []
    labels: List[int] = []
    n = len(samples)
    for i, s in enumerate(samples):
        text = s.get("text") or s.get("content") or ""
        label = int(s.get("label", 0))
        if not text:
            continue
        resp = _http_post_json(f"{llama_url}/embedding",
                               {"content": text}, timeout=60.0)
        if not isinstance(resp, list) or not resp:
            _safe_print(f"  WARN: sample {i+1}/{n} returned empty response")
            continue
        raw = resp[0].get("embedding")
        if isinstance(raw, list) and raw:
            if isinstance(raw[0], list):
                # per-token: mean-pool
                n_tok = len(raw)
                dim = len(raw[0])
                pooled = [0.0] * dim
                for tok in raw:
                    for j, v in enumerate(tok):
                        pooled[j] += v
                pooled = [v / n_tok for v in pooled]
                embeddings.append(pooled)
            else:
                embeddings.append(raw)
            labels.append(label)
        if (i + 1) % 25 == 0 or (i + 1) == n:
            _safe_print(f"  extracted {i+1}/{n} embeddings")
    return {"embeddings": embeddings, "labels": labels}


def _load_training_samples(path: Optional[str]) -> List[Dict]:
    """Load training samples from a JSON or JSONL file.

    Format: list of {"text": str, "label": 0|1} or {"content": str, "label": 0|1}.
    JSONL is detected by .jsonl extension or by leading whitespace check.
    """
    if not path or not os.path.isfile(path):
        return []
    with open(path) as fh:
        content = fh.read()
    if path.endswith(".jsonl") or content.lstrip().startswith("{\""):
        # JSONL
        samples = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(jsonlib.loads(line))
            except jsonlib.JSONDecodeError:
                continue
        return samples
    # JSON array
    try:
        parsed = jsonlib.loads(content)
        if isinstance(parsed, list):
            return parsed
    except jsonlib.JSONDecodeError:
        return []
    return []


def _emit_build(args: argparse.Namespace, color: bool) -> int:
    """Train fresh Lens artifacts for the model llama-server has loaded.

    Doesn't ship its own dataset — users point --samples at a labeled
    JSON/JSONL file (typically pulled from huggingface.co/datasets/itigges22/ATLAS,
    which has the V3 ablation traces with pass/fail labels). Tiny built-in
    sanity datasets are intentionally NOT bundled: a 20-sample C(x) is
    worse than no C(x) (it'll badly mis-rank and the user won't know).
    """
    atlas_root = _atlas_root()

    # 1. Pre-flight: confirm we can probe the model. Reuses the check path
    # so build's UX agrees with check's "this is/isn't compat" verdict.
    _safe_print("[1/4] Probing llama-server…")
    verdict = _check_model(args.model, atlas_root)
    if verdict.verdict == "incompatible":
        _safe_print(f"  {RED if color else ''}Cannot proceed: "
                    f"{verdict.reason}{RESET if color else ''}")
        return 2
    _safe_print(f"  Model emits {verdict.probe.embedding_dim}-dim embeddings "
                f"(model: {verdict.probe.model_name or 'unknown'})")
    if verdict.verdict == "compat" and not args.force:
        _safe_print(f"  {YELL if color else ''}Artifacts already exist at "
                    f"{verdict.artifact_dir} for the current dim. Pass "
                    f"--force to retrain anyway.{RESET if color else ''}")
        return 0

    # 2. Load training data
    _safe_print("[2/4] Loading training samples…")
    if not args.samples:
        _safe_print(f"  {RED if color else ''}--samples PATH is required. "
                    f"Point it at a labeled JSON/JSONL file.{RESET if color else ''}")
        _safe_print("  Format: [{\"text\": str, \"label\": 0|1}, ...]")
        _safe_print("  Pull the canonical training set from:")
        _safe_print("    huggingface.co/datasets/itigges22/ATLAS")
        return 1
    samples = _load_training_samples(args.samples)
    if len(samples) < 50:
        _safe_print(f"  {RED if color else ''}Only {len(samples)} samples "
                    f"loaded. Need >=50 for meaningful training (>=200 "
                    f"recommended).{RESET if color else ''}")
        return 1
    n_pass = sum(1 for s in samples if int(s.get("label", 0)) == 1)
    n_fail = len(samples) - n_pass
    _safe_print(f"  Loaded {len(samples)} samples (PASS={n_pass}, FAIL={n_fail})")
    if n_pass == 0 or n_fail == 0:
        _safe_print(f"  {RED if color else ''}Need both pass and fail "
                    f"samples for contrastive training.{RESET if color else ''}")
        return 1

    # 3. Extract embeddings via /embedding
    _safe_print(f"[3/4] Extracting embeddings via {verdict.probe.url}…")
    start = time.time()
    data = _extract_training_embeddings(samples, verdict.probe.url, color)
    elapsed = time.time() - start
    if not data["embeddings"]:
        _safe_print(f"  {RED if color else ''}No embeddings extracted. "
                    f"Check llama-server logs.{RESET if color else ''}")
        return 1
    _safe_print(f"  Extracted {len(data['embeddings'])} embeddings "
                f"in {elapsed:.1f}s")

    if args.dry_run:
        _safe_print("  (dry-run) skipping training + save")
        return 0

    # 4. Train + save
    _safe_print(f"[4/4] Training CostField "
                f"({args.epochs} epochs, lr={args.lr})…")
    try:
        from geometric_lens.training import train_cost_field, save_cost_field
    except ImportError as e:
        _safe_print(f"  {RED if color else ''}Could not import training "
                    f"module: {e}.{RESET if color else ''}")
        _safe_print("  Make sure you're running from an ATLAS checkout "
                    "(geometric-lens/ must be on PYTHONPATH).")
        return 1
    result = train_cost_field(data, epochs=args.epochs, lr=args.lr,
                              margin=args.margin)
    # train_cost_field returns final_* and best_* keys, not bare test_auc.
    # Surface "best test AUC seen during training" since the final-epoch
    # value can be lower from overfitting.
    test_auc = (result.get("best_test_auc")
                or result.get("final_test_auc") or 0.0)
    train_auc = result.get("final_train_auc") or 0.0
    _safe_print(f"  Train AUC: {train_auc:.4f}  |  Test AUC: {test_auc:.4f} "
                f"(best across epochs)")
    if test_auc < 0.7:
        _safe_print(f"  {YELL if color else ''}Test AUC < 0.70 — model is "
                    f"undertrained. Consider more samples or more "
                    f"epochs.{RESET if color else ''}")

    artifact_dir = args.artifact_dir or verdict.artifact_dir
    os.makedirs(artifact_dir, exist_ok=True)
    cost_path = save_cost_field(result["model"], save_dir=artifact_dir)
    _safe_print(f"  Saved: {cost_path}")
    _safe_print("")
    _safe_print(f"  {GREEN if color else ''}Build complete.{RESET if color else ''}")
    _safe_print(f"  Next: update the registry entry for "
                f"{verdict.matched_model or '<your model>'} to "
                f"lens_status=\"supported\" so doctor + tier surface the win.")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="atlas lens",
        description="Geometric Lens compat probe + build (PC-057, PC-058)")
    sub = parser.add_subparsers(dest="subcommand")

    p_check = sub.add_parser("check",
        help="probe llama-server for Lens compatibility (PC-057)")
    p_check.add_argument("model", nargs="?", default=None,
        help="registry name or path (default: whatever llama-server has loaded)")
    p_check.add_argument("--json", action="store_true",
        help="machine-readable output")
    p_check.add_argument("--no-color", action="store_true")

    p_build = sub.add_parser("build",
        help="train fresh C(x) artifacts for the loaded model (PC-058)")
    p_build.add_argument("model", nargs="?", default=None,
        help="registry name or path (default: whatever llama-server has loaded)")
    p_build.add_argument("--samples", default=None,
        help="path to a labeled JSON/JSONL training file "
             "(format: [{text, label}, ...]; label is 0 or 1)")
    p_build.add_argument("--epochs", type=int, default=200,
        help="training epochs (default: 200)")
    p_build.add_argument("--lr", type=float, default=1e-3,
        help="Adam learning rate (default: 1e-3)")
    p_build.add_argument("--margin", type=float, default=1.0,
        help="contrastive ranking margin (default: 1.0)")
    p_build.add_argument("--artifact-dir", default=None,
        help="where to save cost_field.pt (default: registry-resolved path)")
    p_build.add_argument("--force", action="store_true",
        help="retrain even if compatible artifacts already exist")
    p_build.add_argument("--dry-run", action="store_true",
        help="extract embeddings but skip training + save")
    p_build.add_argument("--no-color", action="store_true")

    args = parser.parse_args(argv)
    if args.subcommand is None:
        parser.print_help()
        return 1

    color = (sys.stdout.isatty()
             and not getattr(args, "no_color", False)
             and not getattr(args, "json", False))

    if args.subcommand == "check":
        return _emit_check(args, color)
    if args.subcommand == "build":
        return _emit_build(args, color)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
