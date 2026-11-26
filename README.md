# LocalAI 3D backends for Apple Silicon

Turn a single image into a textured GLB in LocalAI's 3D Studio. These are three external backends that run locally on an Apple Silicon Mac.

SF3D and SPAR3D are Powered by Stability AI.

| Model | When to use it | What to expect |
| --- | --- | --- |
| **SF3D MPS** (`sf3d-mps`) | Fast previews and trying different source images | About 5 seconds on our torso example with the model loaded. The rear can be poorly reconstructed. |
| **SPAR3D MPS** (`spar3d-mps`) | The back matters and you still want a quick result | About 30 seconds on the same example. It generates a point cloud before reconstruction to improve the shape of hidden surfaces. |
| **TRELLIS Clean** (`trellis-clean`) | Final assets where overall shape and surface quality matter most | Our preferred result of the three. Expect a few minutes. It reconstructs with TRELLIS.2, cleans and reduces the mesh, then bakes colour and normals. |

The timings are from one Mac. The SF3D and SPAR3D comparison used the same torso image, 4,000 triangles and 1024px textures. The preferred model can vary with the image.

## Install on a Mac

Use a native Apple Silicon installation of [LocalAI](https://localai.io/), with 3D Studio and external backends enabled. The package was tested with LocalAI 4.12.1. Start it at least once to create its models and configuration directories. Install [uv](https://docs.astral.sh/uv/) to provide Python 3.12 and install dependencies. For SF3D or SPAR3D, also install Xcode Command Line Tools and Homebrew's `libomp` (`xcode-select --install` and `brew install libomp`). The installer builds the native texture extensions locally.

Clone or unpack this project into a folder you intend to keep. LocalAI stores the launcher's absolute path, so moving the folder after installation breaks that registration.

### SF3D and SPAR3D

Sign in to Hugging Face and accept the terms for the model you want on its model page. Download the files using your own account. You can use the website or the Hugging Face CLI.

- [Stable Fast 3D](https://huggingface.co/stabilityai/stable-fast-3d/tree/main) needs `model.safetensors` and `config.yaml` in the same folder.
- [Stable Point Aware 3D](https://huggingface.co/stabilityai/stable-point-aware-3d/tree/main) needs `model.safetensors`. This project supplies its own reconstruction config.

From the cloned project, run the installer for the model you downloaded. Replace the example path with your weights folder.

```bash
uv run --no-project --python 3.12 install.py sf3d-mps --weights /path/to/sf3d-weights
uv run --no-project --python 3.12 install.py spar3d-mps --weights /path/to/spar3d-weights
```

The installer downloads the U²-Net background-removal model and public image encoder weights. The first generation also has to load the model into memory.

### TRELLIS Clean

Install the native Metal TRELLIS.2 backend and corresponding F16 GGUF weights through LocalAI where available. This package requires a **TRELLIS C library with ABI 11**. One compatible build is the LocalAI Darwin Metal `trellis2cpp` package installed under `~/.localai/backends/metal-trellis2cpp/variants/fallback/` on the development Mac. The installer checks the actual library version. Keep the TRELLIS model and its companion GGUF files together in one directory.

```bash
uv run --no-project --python 3.12 install.py trellis-clean \
  --weights /path/to/trellis-gguf-files \
  --trellis-library /path/to/libtrellis2.dylib
```

If the current TRELLIS package supplies a different ABI, install the tested Darwin build under a separate name. Point `--trellis-library` at `~/.localai/backends/trellis2-abi11/variants/fallback/libtrellis2.dylib`.

```bash
local-ai backends install \
  'oci://quay.io/go-skynet/local-ai-backends@sha256:7c167b8daf531df9a869356d453fba5c1f84e29d96e3aadcaee303c550da603a' \
  trellis2-abi11 --backends-path "$HOME/.localai/backends"
```

The package uses the library directly. Installing `trellis2-4b` from the LocalAI Model Gallery is one way to fetch the companion weights. The installer lists a missing file if the supplied folder has an incomplete set.

### Restart LocalAI

The installer adds the launcher and model configuration to your LocalAI install. It defaults to `~/models/localai` for models and `~/.localai/configuration` for configuration. If your LocalAI instance uses different paths, pass `--models-path` and `--configuration-path`. Restart LocalAI to load the registrations.

If you already use one of these backend IDs, the installer reports a conflict instead of replacing that registration. If installing TRELLIS, allow disk space for intermediate GLBs under `backends/trellis-clean/runs/`. Finished runs can be removed once you have saved the exported GLBs you want.

## Generate a GLB

1. Open **LocalAI → 3D Studio** and upload a clear picture of one object.
2. Select a model. Start with **Auto** quality and **Auto-remove solid background**.
3. Under **Advanced Settings**, leave **Shape steps** blank for a target of **10,000 triangles**. Enter **4** for a target of **4,000 triangles**, or **20** for **20,000**.
4. Select **Generate**, rotate the preview to inspect the rear, and download the GLB.

Triangle reduction is best effort. For API requests, `params: {"target_triangles": "4000"}` takes precedence over Shape steps. In SPAR3D the point diffusion still runs 32 steps even when you request a smaller mesh.

Leave **Material steps** blank to use the default textures. The defaults are 1024px for SF3D and SPAR3D and 2048px for TRELLIS Clean. SF3D and SPAR3D accept only Auto quality and require Guidance and Seed to be blank. TRELLIS Clean also accepts 512 and 1024 quality. This changes reconstruction resolution separately from the output texture size.

## Source and weights

- The SF3D and SPAR3D source directories retain their upstream Stability AI licence files. Their model checkpoints are downloaded separately under the upstream terms.
- The TRELLIS Clean adapter requires a separately installed TRELLIS native runtime and weights. See [LocalAI's TRELLIS.2 backend](https://github.com/localai-org/trellis2cpp).
- The LocalAI gRPC bindings are generated from LocalAI's backend protocol. Python packages and native dependencies are installed separately.
