"""Tests for atlas.cli.commands.init (PC-054) — the first-run install wizard.

The wizard is a thin composer over tier + model_registry + model.install,
so these tests focus on the wizard's own behavior:

  - --yes happy path writes .env + api-keys.json with expected values
  - --reconfigure backs up existing .env before overwriting
  - already-configured guard refuses without --reconfigure
  - --skip-download produces config without calling install
  - --dry-run touches no files
  - --json shape is stable for the bootstrap script
  - api-keys.json permissions are 0600 + parent 0700
  - reconfigure of api-keys.json backs up the existing one
"""

import json
import os
import pathlib
import stat

import pytest

from atlas.cli.commands import init, tier


# The wizard refuses on cpu tier (PC-054) — correct production behavior,
# but it means the happy-path tests below would all return rc=1 on a
# CPU-only host (e.g. GitHub runners). This autouse fixture mocks the
# probe to a GPU-equipped host so tests proceed past the GPU guard.
# The explicit cpu-refusal test (test_refuses_on_cpu_tier) overrides
# this with its own monkeypatch.
@pytest.fixture(autouse=True)
def _mock_gpu_probe(monkeypatch):
    gpu_probe = tier.Probe(
        has_gpu=True, gpu_name="NVIDIA Test GPU", vram_gb=24.0, gpu_count=1,
        system_ram_gb=64.0, cpu_cores=16, disk_free_gb=500.0, platform="linux")
    monkeypatch.setattr(tier, "probe", lambda install_dir=None: gpu_probe)


def _make_atlas_root(tmp_path) -> str:
    """Create a fake atlas_root with a docker-compose.yml so _find_atlas_root
    walks up from tmp_path and lands here. Returns absolute path."""
    root = tmp_path / "atlas_root"
    root.mkdir()
    (root / "docker-compose.yml").write_text("# fake compose for tests\n")
    return str(root)


def _run(monkeypatch, atlas_root, argv):
    """Run init.main with CWD set to atlas_root so atlas_root resolution
    finds the fake docker-compose.yml."""
    monkeypatch.chdir(atlas_root)
    return init.main(argv)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_yes_skip_download_writes_env_and_keys(tmp_path, monkeypatch, capsys):
    root = _make_atlas_root(tmp_path)
    rc = _run(monkeypatch, root,
              ["--yes", "--skip-download", "--no-color"])
    assert rc == 0

    env_path = os.path.join(root, ".env")
    keys_path = os.path.join(root, "secrets", "api-keys.json")
    assert os.path.isfile(env_path)
    assert os.path.isfile(keys_path)

    body = pathlib.Path(env_path).read_text()
    # Wizard must write every key the compose stack reads at boot.
    for key in ("ATLAS_MODELS_DIR", "ATLAS_MODEL_FILE", "ATLAS_MODEL_NAME",
                "ATLAS_CTX_SIZE", "ATLAS_GHCR_OWNER", "ATLAS_IMAGE_TAG",
                "ATLAS_LLAMA_PORT", "PARALLEL_SLOTS",
                "ATLAS_BACKEND", "ATLAS_GPU_VENDOR", "ATLAS_GPU_INDEX"):
        assert f"{key}=" in body, f"missing {key} in .env"

    # Default models_dir is ./models when it equals atlas_root/models.
    assert "ATLAS_MODELS_DIR=./models" in body
    assert "ATLAS_IMAGE_TAG=latest" in body
    assert "ATLAS_GHCR_OWNER=itigges22" in body


def test_api_keys_file_has_strict_permissions(tmp_path, monkeypatch):
    root = _make_atlas_root(tmp_path)
    rc = _run(monkeypatch, root,
              ["--yes", "--skip-download", "--no-color"])
    assert rc == 0

    keys_dir = os.path.join(root, "secrets")
    keys_path = os.path.join(keys_dir, "api-keys.json")
    dir_mode = stat.S_IMODE(os.stat(keys_dir).st_mode)
    file_mode = stat.S_IMODE(os.stat(keys_path).st_mode)
    assert dir_mode == 0o700, f"secrets/ mode {oct(dir_mode)} != 0700"
    assert file_mode == 0o600, f"api-keys.json mode {oct(file_mode)} != 0600"


def test_api_keys_payload_shape(tmp_path, monkeypatch):
    root = _make_atlas_root(tmp_path)
    rc = _run(monkeypatch, root,
              ["--yes", "--skip-download", "--no-color"])
    assert rc == 0
    payload = json.loads(pathlib.Path(root, "secrets", "api-keys.json").read_text())
    # Exactly one key, of the expected sk-atlas-* prefix, valued correctly.
    assert len(payload) == 1
    (key, value), = payload.items()
    assert key.startswith("sk-atlas-")
    assert len(key) > len("sk-atlas-") + 20  # token_urlsafe(32) is well over 20 chars
    assert value == {"user": "local", "created_by": "atlas init"}


# ---------------------------------------------------------------------------
# Already-configured guard + --reconfigure
# ---------------------------------------------------------------------------

def test_already_configured_refuses_without_reconfigure(tmp_path, monkeypatch, capsys):
    root = _make_atlas_root(tmp_path)
    # Pre-existing .env from an earlier setup.
    existing_env = os.path.join(root, ".env")
    pathlib.Path(existing_env).write_text("ATLAS_MODEL_FILE=hand-edited.gguf\n")

    rc = _run(monkeypatch, root,
              ["--yes", "--skip-download", "--no-color"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "Already configured" in out
    assert "--reconfigure" in out
    # And critically — original .env was NOT modified.
    # Path.read_text() closes the handle; bare open().read() in an
    # assert leaks the descriptor and can leak file state under -O.
    assert pathlib.Path(existing_env).read_text() == \
        "ATLAS_MODEL_FILE=hand-edited.gguf\n"


def test_reconfigure_backs_up_existing_env(tmp_path, monkeypatch, capsys):
    root = _make_atlas_root(tmp_path)
    existing_env = os.path.join(root, ".env")
    pathlib.Path(existing_env).write_text("ATLAS_MODEL_FILE=old.gguf\n")

    rc = _run(monkeypatch, root,
              ["--yes", "--skip-download", "--no-color", "--reconfigure"])
    assert rc == 0
    backup = existing_env + ".bak"
    assert os.path.isfile(backup)
    assert pathlib.Path(backup).read_text() == "ATLAS_MODEL_FILE=old.gguf\n"
    # New .env is the wizard's render — has the structured comment header.
    new_body = pathlib.Path(existing_env).read_text()
    assert "generated by `atlas init`" in new_body


def test_reconfigure_without_existing_env_still_works(tmp_path, monkeypatch):
    root = _make_atlas_root(tmp_path)
    rc = _run(monkeypatch, root,
              ["--yes", "--skip-download", "--no-color", "--reconfigure"])
    assert rc == 0
    assert os.path.isfile(os.path.join(root, ".env"))
    # No backup file when there was nothing to back up.
    assert not os.path.isfile(os.path.join(root, ".env.bak"))


def test_reconfigure_backs_up_existing_api_keys(tmp_path, monkeypatch):
    root = _make_atlas_root(tmp_path)
    secrets_dir = os.path.join(root, "secrets")
    os.makedirs(secrets_dir, mode=0o700)
    keys_path = os.path.join(secrets_dir, "api-keys.json")
    pathlib.Path(keys_path).write_text('{"sk-old-key": {"user": "alice"}}\n')
    os.chmod(keys_path, 0o600)

    # Need .env present too so the reconfigure flag is the actual gate.
    pathlib.Path(root, ".env").write_text("ATLAS_MODEL_FILE=foo.gguf\n")

    rc = _run(monkeypatch, root,
              ["--yes", "--skip-download", "--no-color", "--reconfigure"])
    assert rc == 0
    bak = keys_path + ".bak"
    assert os.path.isfile(bak)
    assert "sk-old-key" in pathlib.Path(bak).read_text()
    # New file has a fresh sk-atlas-* key, not the old one.
    new = json.loads(pathlib.Path(keys_path).read_text())
    assert all(k.startswith("sk-atlas-") for k in new.keys())


# ---------------------------------------------------------------------------
# --skip-download + --dry-run
# ---------------------------------------------------------------------------

def test_skip_download_does_not_touch_models_dir(tmp_path, monkeypatch, capsys):
    root = _make_atlas_root(tmp_path)
    rc = _run(monkeypatch, root,
              ["--yes", "--skip-download", "--no-color"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Skipping download" in out
    # Models dir not auto-created when skipped — the user is responsible
    # for placing the gguf themselves.
    assert not os.path.exists(os.path.join(root, "models"))


def test_dry_run_touches_no_files(tmp_path, monkeypatch, capsys):
    root = _make_atlas_root(tmp_path)
    rc = _run(monkeypatch, root,
              ["--yes", "--skip-download", "--dry-run", "--no-color"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "(dry-run)" in out
    # Nothing on disk except the seed compose file.
    assert os.listdir(root) == ["docker-compose.yml"]


# ---------------------------------------------------------------------------
# --json shape
# ---------------------------------------------------------------------------

def test_json_output_shape_is_stable(tmp_path, monkeypatch, capsys):
    root = _make_atlas_root(tmp_path)
    rc = _run(monkeypatch, root,
              ["--yes", "--skip-download", "--no-color", "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    # JSON object is at the very end of stdout — find its opening brace.
    blob = out[out.rindex("{"):]
    payload = json.loads(blob)
    expected_keys = {"atlas_root", "tier", "model", "models_dir",
                     "env_path", "env_backup", "api_keys_path",
                     "api_keys_backup", "api_key", "image_tag",
                     "ghcr_owner", "dry_run", "backend", "gpu"}
    assert expected_keys.issubset(payload.keys())
    assert payload["atlas_root"] == root
    assert payload["dry_run"] is False
    # api_key in JSON matches the file we wrote.
    file_payload = json.loads(pathlib.Path(payload["api_keys_path"]).read_text())
    assert payload["api_key"] in file_payload


def test_amd_probe_renders_rocm_backend(tmp_path, monkeypatch, capsys):
    """When the probe reports an AMD GPU, .env carries ATLAS_BACKEND=rocm
    and the ROCm compose-override hint appears."""
    amd_gpu = tier.GPUInfo(vendor="amd", name="AMD Radeon RX 7900 XTX",
                            vram_gb=24.0, compute_target="gfx1100", index=0)
    amd_probe = tier.Probe(
        has_gpu=True, gpu_name=amd_gpu.name, gpu_vendor="amd",
        vram_gb=24.0, gpu_count=1, gpus=[amd_gpu],
        system_ram_gb=32.0, cpu_cores=8, disk_free_gb=200.0, platform="linux")
    monkeypatch.setattr(tier, "probe", lambda install_dir=None: amd_probe)

    root = _make_atlas_root(tmp_path)
    rc = _run(monkeypatch, root,
              ["--yes", "--skip-download", "--no-color"])
    assert rc == 0
    body = pathlib.Path(root, ".env").read_text()
    assert "ATLAS_BACKEND=rocm" in body
    assert "ATLAS_GPU_VENDOR=amd" in body
    assert "ATLAS_GPU_INDEX=0" in body
    # The compose-override hint comment should be present so users know
    # the second compose file is required for ROCm.
    assert "docker-compose.rocm.yml" in body


def test_apple_silicon_probe_offers_vulkan_fallback(tmp_path, monkeypatch, capsys):
    """Apple Silicon hosts get the Vulkan-via-MoltenVK fallback offered
    when Metal isn't packaged (PC-114). The Metal-is-V3.1.2 message
    still appears so the user knows the fast path exists, but the
    wizard no longer hard-refuses — Vulkan-on-darwin lets them at
    least boot the stack via Docker."""
    apple_gpu = tier.GPUInfo(vendor="apple", name="Apple M3 Pro",
                              vram_gb=18.0, compute_target=None, index=0)
    apple_probe = tier.Probe(
        has_gpu=True, gpu_name=apple_gpu.name, gpu_vendor="apple",
        vram_gb=18.0, gpu_count=1, gpus=[apple_gpu],
        system_ram_gb=18.0, cpu_cores=11, disk_free_gb=200.0, platform="darwin")
    monkeypatch.setattr(tier, "probe", lambda install_dir=None: apple_probe)
    # Force vulkan_available() True on this fake darwin host — the
    # production tier.vulkan_available checks sys.platform but we're
    # running these tests on Linux.
    monkeypatch.setattr(tier, "vulkan_available", lambda: True)

    root = _make_atlas_root(tmp_path)
    rc = _run(monkeypatch, root,
              ["--yes", "--skip-download", "--no-color"])
    out = capsys.readouterr().out
    # Metal-is-V3.1.2 message still emitted so the user knows the fast
    # native path exists (the slow Vulkan-via-MoltenVK fallback isn't
    # the recommended end state for Mac users).
    assert "Metal" in out
    assert "V3.1.2" in out
    # New behavior — --yes accepts the Vulkan fallback offer.
    assert "Vulkan" in out
    assert rc == 0
    body = pathlib.Path(root, ".env").read_text()
    assert "ATLAS_BACKEND=vulkan" in body


def test_apple_silicon_refuses_when_vulkan_fallback_declined(
    tmp_path, monkeypatch, capsys
):
    """When the user says no to the Vulkan fallback offer, the wizard
    still refuses (writes nothing) rather than proceeding with an
    unsupported backend. This is the explicit-decline path."""
    apple_gpu = tier.GPUInfo(vendor="apple", name="Apple M3 Pro",
                              vram_gb=18.0, compute_target=None, index=0)
    apple_probe = tier.Probe(
        has_gpu=True, gpu_name=apple_gpu.name, gpu_vendor="apple",
        vram_gb=18.0, gpu_count=1, gpus=[apple_gpu],
        system_ram_gb=18.0, cpu_cores=11, disk_free_gb=200.0, platform="darwin")
    monkeypatch.setattr(tier, "probe", lambda install_dir=None: apple_probe)
    monkeypatch.setattr(tier, "vulkan_available", lambda: True)
    # Interactive mode but stdin returns "n" to the Vulkan prompt.
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *a, **kw: "n")

    root = _make_atlas_root(tmp_path)
    # Note: NO --yes here so the prompt actually runs and gets the "n".
    rc = _run(monkeypatch, root,
              ["--skip-download", "--no-color"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "Vulkan" in out
    # .env must NOT be written when the user explicitly declines.
    assert not os.path.isfile(os.path.join(root, ".env"))


def test_apple_silicon_refuses_when_vulkan_unavailable(
    tmp_path, monkeypatch, capsys
):
    """Edge: Apple Silicon + vulkan_available()=False (e.g. some weird
    container build that strips MoltenVK) → no fallback exists, wizard
    must refuse rather than write a broken .env."""
    apple_gpu = tier.GPUInfo(vendor="apple", name="Apple M3 Pro",
                              vram_gb=18.0, compute_target=None, index=0)
    apple_probe = tier.Probe(
        has_gpu=True, gpu_name=apple_gpu.name, gpu_vendor="apple",
        vram_gb=18.0, gpu_count=1, gpus=[apple_gpu],
        system_ram_gb=18.0, cpu_cores=11, disk_free_gb=200.0, platform="darwin")
    monkeypatch.setattr(tier, "probe", lambda install_dir=None: apple_probe)
    monkeypatch.setattr(tier, "vulkan_available", lambda: False)

    root = _make_atlas_root(tmp_path)
    rc = _run(monkeypatch, root,
              ["--yes", "--skip-download", "--no-color"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "Vulkan fallback isn't available" in out
    assert not os.path.isfile(os.path.join(root, ".env"))


# ---------------------------------------------------------------------------
# --image-tag / --ghcr-owner overrides
# ---------------------------------------------------------------------------

def test_image_tag_override_lands_in_env(tmp_path, monkeypatch):
    root = _make_atlas_root(tmp_path)
    rc = _run(monkeypatch, root,
              ["--yes", "--skip-download", "--no-color",
               "--image-tag", "v1.2.3", "--ghcr-owner", "myfork"])
    assert rc == 0
    body = pathlib.Path(root, ".env").read_text()
    assert "ATLAS_IMAGE_TAG=v1.2.3" in body
    assert "ATLAS_GHCR_OWNER=myfork" in body


# ---------------------------------------------------------------------------
# --models-dir override
# ---------------------------------------------------------------------------

def test_models_dir_override_lands_in_env_as_absolute(tmp_path, monkeypatch):
    root = _make_atlas_root(tmp_path)
    custom = tmp_path / "custom_models"
    custom.mkdir()
    rc = _run(monkeypatch, root,
              ["--yes", "--skip-download", "--no-color",
               "--models-dir", str(custom)])
    assert rc == 0
    body = pathlib.Path(root, ".env").read_text()
    # Non-default path is written verbatim (absolute), not as ./models.
    assert f"ATLAS_MODELS_DIR={custom}" in body
    assert "ATLAS_MODELS_DIR=./models" not in body


# ---------------------------------------------------------------------------
# PC-054 audit fixes
# ---------------------------------------------------------------------------

def test_refuses_when_no_atlas_checkout_in_cwd(tmp_path, monkeypatch, capsys):
    """Running outside an ATLAS checkout (no docker-compose.yml in CWD or
    any parent) refuses up-front rather than silently writing .env into
    a random directory."""
    # tmp_path has no docker-compose.yml — and pytest's tmp_path is under
    # /tmp/pytest-of-*/, none of whose ancestors have one either.
    monkeypatch.chdir(tmp_path)
    rc = init.main(["--yes", "--skip-download", "--no-color"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "no docker-compose.yml found" in out
    # And critically — nothing got written.
    assert os.listdir(tmp_path) == []


def test_refuses_on_cpu_tier(tmp_path, monkeypatch, capsys):
    """When tier.classify returns 'cpu' (no GPU), refuse rather than
    silently recommend a 16GB-VRAM model the user can't run."""
    root = _make_atlas_root(tmp_path)

    # Force a cpu probe regardless of the actual host.
    from atlas.cli.commands import tier
    cpu_probe = tier.Probe(
        has_gpu=False, gpu_name=None, vram_gb=0.0, gpu_count=0,
        system_ram_gb=8.0, cpu_cores=4, disk_free_gb=100.0, platform="linux")
    monkeypatch.setattr(tier, "probe", lambda install_dir=None: cpu_probe)

    rc = _run(monkeypatch, root,
              ["--yes", "--skip-download", "--no-color"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "No GPU detected" in out
    assert "requires a CUDA, ROCm, or Metal-capable GPU" in out
    # Nothing got written — the refusal happens before step 4.
    assert not os.path.isfile(os.path.join(root, ".env"))
    assert not os.path.isdir(os.path.join(root, "secrets"))


def test_yes_skips_prompts(tmp_path, monkeypatch, capsys):
    """--yes makes the wizard non-interactive — input() is never called.
    Patch input() to raise so the test fails loudly if the wizard
    accidentally tries to prompt."""
    root = _make_atlas_root(tmp_path)

    def _no_prompt(*args, **kwargs):
        raise AssertionError("wizard called input() despite --yes")
    monkeypatch.setattr("builtins.input", _no_prompt)

    rc = _run(monkeypatch, root,
              ["--yes", "--skip-download", "--no-color"])
    assert rc == 0


def test_interactive_decline_aborts_wizard(tmp_path, monkeypatch, capsys):
    """When the user answers 'n' at the model-confirmation prompt,
    the wizard exits 1 cleanly and writes nothing."""
    root = _make_atlas_root(tmp_path)

    # Force interactive mode: claim stdin is a TTY, no --yes.
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    # First prompt: decline.
    monkeypatch.setattr("builtins.input", lambda *a, **kw: "n")

    rc = _run(monkeypatch, root,
              ["--skip-download", "--no-color"])
    assert rc == 1
    # No .env, no secrets/.
    assert not os.path.isfile(os.path.join(root, ".env"))
    assert not os.path.isdir(os.path.join(root, "secrets"))


def test_interactive_default_yes_on_empty_input(tmp_path, monkeypatch):
    """Pressing Enter at a [Y/n] prompt accepts the default (yes)."""
    root = _make_atlas_root(tmp_path)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    # Empty string = pressed Enter without typing anything.
    monkeypatch.setattr("builtins.input", lambda *a, **kw: "")

    rc = _run(monkeypatch, root,
              ["--skip-download", "--no-color"])
    assert rc == 0
    assert os.path.isfile(os.path.join(root, ".env"))


def test_non_tty_stdin_is_treated_as_yes(tmp_path, monkeypatch):
    """When stdin isn't a TTY (piped input, CI), the wizard never
    prompts — it uses defaults the same way --yes does. Otherwise CI
    runs would hang waiting on input that will never come."""
    root = _make_atlas_root(tmp_path)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    # input() being callable would still be a bug — verify it isn't called.
    def _no_prompt(*args, **kwargs):
        raise AssertionError("wizard called input() with non-TTY stdin")
    monkeypatch.setattr("builtins.input", _no_prompt)

    rc = _run(monkeypatch, root,
              ["--skip-download", "--no-color"])
    assert rc == 0
    assert os.path.isfile(os.path.join(root, ".env"))


# ---------------------------------------------------------------------------
# Vulkan universal backend (PC-114, #114)
# ---------------------------------------------------------------------------

def test_backend_vulkan_override_writes_vulkan_into_env(tmp_path, monkeypatch):
    """--backend vulkan forces ATLAS_BACKEND=vulkan in .env regardless of
    detected GPU vendor. Lets a user opt into the universal fallback
    even on a CUDA-capable box (e.g. to smoke-test the Vulkan path)."""
    root = _make_atlas_root(tmp_path)
    rc = _run(monkeypatch, root,
              ["--yes", "--skip-download", "--no-color",
               "--backend", "vulkan"])
    assert rc == 0
    body = pathlib.Path(root, ".env").read_text()
    assert "ATLAS_BACKEND=vulkan" in body
    # The vulkan-specific compose hint should appear in the header so
    # operators don't accidentally bring the stack up with only the base
    # file (which would route llama-server to the CUDA image).
    assert "docker-compose.vulkan.yml" in body


def test_backend_vulkan_override_skips_vendor_compat_check(tmp_path, monkeypatch):
    """The --backend override short-circuits the vendor-isn't-packaged
    refusal path. Without this, --backend vulkan would still bail on a
    host with an Intel iGPU (since intel maps to the SYCL backend that
    isn't packaged yet)."""
    root = _make_atlas_root(tmp_path)
    # Simulate an Intel iGPU (intel → sycl → not packaged → would normally refuse).
    intel_probe = tier.Probe(
        has_gpu=True, gpu_name="Intel Arc A770", vram_gb=16.0, gpu_count=1,
        system_ram_gb=64.0, cpu_cores=16, disk_free_gb=500.0, platform="linux",
        gpu_vendor="intel")
    monkeypatch.setattr(tier, "probe", lambda install_dir=None: intel_probe)

    rc = _run(monkeypatch, root,
              ["--yes", "--skip-download", "--no-color",
               "--backend", "vulkan"])
    assert rc == 0
    body = pathlib.Path(root, ".env").read_text()
    assert "ATLAS_BACKEND=vulkan" in body


def test_vulkan_available_returns_true_with_vulkaninfo_on_path(monkeypatch):
    """tier.vulkan_available() should return True when vulkaninfo is
    found on PATH — the cheapest reliable signal that the host can
    run Vulkan."""
    monkeypatch.setattr("shutil.which",
                        lambda name: "/usr/bin/vulkaninfo" if name == "vulkaninfo" else None)
    assert tier.vulkan_available() is True


def test_vulkan_available_returns_true_with_dev_dri(monkeypatch):
    """Even without vulkaninfo on the host, /dev/dri presence means the
    Vulkan-in-container path will work (Mesa ICDs inside the image
    handle the rest)."""
    monkeypatch.setattr("shutil.which", lambda name: None)
    monkeypatch.setattr("os.path.exists",
                        lambda p: True if p == "/dev/dri" else False)
    assert tier.vulkan_available() is True


def test_vulkan_available_returns_true_on_darwin(monkeypatch):
    """macOS hosts get True regardless (MoltenVK + Docker Desktop on Mac
    provides the Vulkan path even with no native vulkaninfo)."""
    monkeypatch.setattr("shutil.which", lambda name: None)
    monkeypatch.setattr("os.path.exists", lambda p: False)
    monkeypatch.setattr(tier, "detect_gpu", lambda: [])
    monkeypatch.setattr("sys.platform", "darwin")
    assert tier.vulkan_available() is True


def test_vulkan_available_false_when_no_signal(monkeypatch):
    """No vulkaninfo, no /dev/dri, not macOS, no detected GPU →
    Vulkan probably won't work even via lavapipe. Return False so the
    wizard refuses cleanly instead of writing a doomed .env."""
    monkeypatch.setattr("shutil.which", lambda name: None)
    monkeypatch.setattr("os.path.exists", lambda p: False)
    monkeypatch.setattr(tier, "detect_gpu", lambda: [])
    monkeypatch.setattr("sys.platform", "linux")
    assert tier.vulkan_available() is False
