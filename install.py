"""Install an external backend into a native LocalAI instance on Apple Silicon."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / ".runtime"
PYTHON = ROOT / ".venv/bin/python"
MODELS = ("sf3d-mps", "spar3d-mps", "trellis-clean")
TRELLIS_FILES = (
    "dino", "ss_flow", "ss_dec", "slat_flow", "slat_flow_1024", "shape_dec",
    "shape_enc", "tex_dec", "tex_slat_flow_512", "tex_slat_flow_1024",
)


def require_file(path):
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"Missing or empty file: {path}")


def write_atomic(path, content, mode=0o600):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        try:
            file.write(content.encode())
            file.close()
            temporary.chmod(mode)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def install_dependencies(model):
    if not PYTHON.is_file():
        subprocess.run(["uv", "venv", "--python", "3.12", str(ROOT / ".venv")], check=True)
    pip = ["uv", "pip", "install", "--python", str(PYTHON)]
    requirements = ["requirements.txt"]
    if model == "trellis-clean":
        requirements.append("requirements-trellis.txt")
    else:
        requirements.append("requirements-mps.txt")
        if model == "spar3d-mps":
            requirements.append("requirements-spar3d.txt")
    subprocess.run(pip + [arg for file in requirements for arg in ("-r", str(ROOT / file))], check=True)
    if model == "trellis-clean":
        return
    omp = Path(subprocess.check_output(["brew", "--prefix", "libomp"], text=True).strip()) / "lib"
    require_file(omp / "libomp.dylib")
    env = dict(os.environ, USE_METAL="1", USE_CUDA="0", LIBRARY_PATH=str(omp),
               LDFLAGS=f"-L{omp} -Wl,-rpath,{omp}",
               PATH=f"{PYTHON.parent}:{os.environ['PATH']}")
    subprocess.run(pip + ["--no-build-isolation", str(ROOT / "shared/texture_baker"),
                          str(ROOT / "shared/uv_unwrapper")], env=env, check=True)
    subprocess.run([str(PYTHON), "-c",
                    "import rembg, torch;\nif not torch.backends.mps.is_available(): raise RuntimeError('MPS unavailable')\n"
                    "rembg.new_session('u2net', providers=['CPUExecutionProvider'])"],
                   env=dict(os.environ, U2NET_HOME=str(RUNTIME / "rembg")), check=True)

    public_weights = [
        ("facebook/dinov2-large", "config.json"),
        ("facebook/dinov2-large", "model.safetensors"),
    ]
    if model == "sf3d-mps":
        public_weights.append(("laion/CLIP-ViT-B-32-laion2B-s34B-b79K", "open_clip_pytorch_model.bin"))
    subprocess.run([str(PYTHON), "-c",
                    "import json,sys; from huggingface_hub import hf_hub_download; "
                    "[hf_hub_download(repo, name, token=False) for repo, name in json.loads(sys.argv[1])]",
                    json.dumps(public_weights)],
                   env=dict(os.environ, HF_HUB_CACHE=str(RUNTIME / "huggingface")), check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--models-path", type=Path, default=Path.home() / "models/localai")
    parser.add_argument("--configuration-path", type=Path, default=Path.home() / ".localai/configuration")
    parser.add_argument("--trellis-library", type=Path)
    args = parser.parse_args()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        parser.error("Native Apple Silicon macOS is required")

    weights = args.weights.expanduser().resolve(strict=True)
    models = args.models_path.expanduser().resolve(strict=True)
    configuration = args.configuration_path.expanduser().resolve(strict=True)
    if not all(path.is_dir() for path in (weights, models, configuration)):
        parser.error("Weights, models, and configuration paths must be directories")

    model = args.model
    library = None
    if model == "trellis-clean":
        if args.trellis_library is None:
            parser.error("trellis-clean requires --trellis-library pointing to libtrellis2.dylib (ABI 11)")
        library = args.trellis_library.expanduser().resolve(strict=True)
        require_file(library)
        for name in TRELLIS_FILES:
            require_file(weights / f"{name}_f16.gguf")
    else:
        if args.trellis_library is not None:
            parser.error("--trellis-library is only valid for trellis-clean")
        require_file(weights / "model.safetensors")
        if model == "sf3d-mps":
            require_file(weights / "config.yaml")

    model_dir = models / (model + "-weights")
    config = models / (model + ".yaml")
    config_text = (ROOT / "models" / config.name).read_text()
    launcher = RUNTIME / (model + ".sh")
    registry = configuration / "external_backends.json"
    RUNTIME.mkdir(exist_ok=True)
    with (configuration / ".localai-3d-install.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        mapping = json.loads(registry.read_text()) if registry.exists() else {}
        if not isinstance(mapping, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in mapping.items()):
            parser.error(f"Invalid backend registry: {registry}")
        if model in mapping and mapping[model] != str(launcher):
            parser.error(f"Backend {model} is already registered at {mapping[model]}")
        if config.exists() and config.read_text() != config_text:
            parser.error(f"Different model configuration already exists: {config}")
        if model_dir.is_symlink():
            if model_dir.resolve(strict=True) != weights:
                parser.error(f"Weights link points elsewhere: {model_dir}")
        elif model_dir.exists():
            parser.error(f"Weights path already exists: {model_dir}")

        install_dependencies(model)
        if library is not None:
            subprocess.run([str(PYTHON), "-c",
                            "import ctypes,sys; lib=ctypes.CDLL(sys.argv[1]); "
                            "abi=lib.t2_abi_version()\n"
                            "if abi != 11: raise RuntimeError(f'Expected TRELLIS ABI 11, found {abi}')",
                            str(library)], env=dict(os.environ, DYLD_LIBRARY_PATH=str(library.parent)), check=True)
            write_atomic(RUNTIME / "trellis-library", str(library) + "\n")
        write_atomic(launcher, f'#!/bin/sh\nexec {shlex.quote(str(ROOT / "run.sh"))} {model} "$@"\n', 0o700)
        if not model_dir.is_symlink():
            model_dir.symlink_to(weights, target_is_directory=True)
        mapping[model] = str(launcher)
        write_atomic(registry, json.dumps(mapping, indent=2) + "\n")
        write_atomic(config, config_text)
    print(f"Installed {model}. Restart LocalAI, then select {model} in 3D Studio.")


if __name__ == "__main__":
    main()
