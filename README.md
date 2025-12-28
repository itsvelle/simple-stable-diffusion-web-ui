# SDUI (SDXL UI)

A simple local web UI for SDXL image generation using **FastAPI + Gradio**.

- Python **3.12+**
- Runs on Linux CPU, NVIDIA CUDA, and **AMD ROCm** (PyTorch ROCm)
- Loads *only* from local files in `models/`
- Saves outputs to `outputs/` and serves them at `/outputs/...`

## Folder layout

- `models/base/`   SDXL base model (Diffusers folder or `.safetensors`)
- `models/refiner/` (optional) SDXL refiner model
- `models/vae/`    (optional) VAE
- `models/lora/`   (optional) LoRA(s)
- `outputs/`       generated images

Create the missing folders you need (only `models/base` is required).

## Install

Create a venv and install dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### PyTorch for AMD ROCm (Linux)

Install a ROCm-enabled PyTorch build appropriate for your distro + ROCm version.

- Start here: <https://pytorch.org/get-started/locally/>
- For ROCm wheels, PyTorch typically uses an index like `https://download.pytorch.org/whl/rocm*`

After installing ROCm PyTorch, `torch.cuda.is_available()` should return `True` (ROCm uses the `cuda` device string).

## Add your models

Put your model files under `models/`.

Examples:

- Diffusers SDXL base:
  - `models/base/stabilityai--stable-diffusion-xl-base-1.0/` (contains `model_index.json`)
- Single-file SDXL:
  - `models/base/sdxl_base.safetensors`
- LoRA:
  - `models/lora/my_style.safetensors`
- VAE:
  - `models/vae/sdxl_vae.safetensors` (or a diffusers VAE folder)

## Run

```bash
python run.py
```

Then open:

- UI: <http://localhost:7860/>
- Health: <http://localhost:7860/api/health>

## Notes

- If you add new model files while the server is running, restart to refresh dropdowns.
- If you get out-of-memory errors, reduce resolution, steps, number of images, or use fewer LoRAs.

## Generation history (SQLite)

Each generation is saved to a local SQLite DB (default: `outputs/history.sqlite3`) including:

- Prompt + negative prompt
- Core settings (steps/CFG/scheduler/prediction type/etc.)
- Output filenames saved under `outputs/`

In the UI, open the **History** section to browse recent items and click **Load settings into form** to reuse prompts/settings.

## NoobAI-XL (v-pred models)

NoobAI-XL has both epsilon/noise-pred and **v-prediction** variants.

- For most epsilon/noise-pred models, defaults are fine.
- For **v-pred** models, set **Prediction type** to `v_prediction`.
- Recommended starting point: `Euler` or `Euler a`, CFG ~3–5, steps ~32–40.
- If you see oversaturation/overexposure, try **CFG rescale (guidance_rescale)** around `0.7` and keep `rescale_betas_zero_snr` enabled.
