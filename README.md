---
base_model:
- TheyCallMeHex/LCM-Dreamshaper-V7-ONNX
tags:
- rknn
- LCM
- stable-diffusion
---
# LCM Super-Resolution Server (RKNN)

This repository adds RKNN-accelerated image super-resolution as a server-side capability that integrates cleanly with an LCM Stable Diffusion deployment on Rockchip NPUs (RK3588 / RK356x).

Super-resolution is exposed both as:
  * a standalone REST endpoint, and

  * an optional post-process step inside /generate.

The design follows the same safety guarantees as the LCM server:
  * one RKNN runtime per worker

  * no shared RKNN objects across threads

  * queue backpressure

  * clean startup / shutdown lifecycle

What this is based on

This work is directly based on the original standalone demo repository:

Original repo:
https://github.com/Mario5Gray/rknn-superresolution

That project demonstrates:
  * RKNNLite-based inference on Rockchip NPU

  * YCbCr color space processing

  * Super-resolution applied to the luminance (Y) channel

  * Tiled inference for arbitrary image sizes

  * Reconstruction with bicubic chroma upscaling

The server implementation preserves the same algorithmic approach, but wraps it in:
  * a FastAPI server

  * worker-owned RKNN runtimes

  * request queuing

  * optional post-processing inside Stable Diffusion generatio

⚠️ Important:
As with the original repo, the SR model is not included. You must supply super-resolution-10.rknn yourself.

## Model requirements (not included)

The server expects an RKNN-converted super-resolution model.

### Original source model
  * super-resolution-10.onnx

  * From the ONNX model zoo (Sub-Pixel CNN 2016)

ONNX model source:
https://github.com/onnx/models/tree/main/validated/vision/super_resolution/sub_pixel_cnn_2016/model

### Conversion steps (same as original repo)
  1.  Download super-resolution-10.onnx
  2.  Run convert.py on a PC with RKNN Toolkit installed
  3.  Set target_platform to match your board (e.g. rk3588)
  4.  Copy the generated super-resolution-10.rknn to your device

The server will not start if the SR model path is invalid.

## Server deployment

### Environment variables

```bash
# Enable / disable SR
export SR_ENABLED=1

# Path to the RKNN SR model (must exist)
export SR_MODEL_PATH=/models/super-resolution-10.rknn

# SR tiling parameters
export SR_INPUT_SIZE=224
export SR_OUTPUT_SIZE=672

# SR worker configuration
export SR_NUM_WORKERS=1
export SR_QUEUE_MAX=32
export SR_MAX_PIXELS=12000000
export SR_REQUEST_TIMEOUT=120
```

### Start the Server

```bash
python lcm_sr_server.py
```

Or with uvicorn explicitly:

```bash
uvicorn lcm_sr_server:app \
  --host 0.0.0.0 \
  --port 4200
```

On startup, you should see logs indicating that:
  * SR workers are created

  * each worker loads super-resolution-10.rknn

If the model cannot be found, the server will fail fast.

## Client usage

1. Standalone super-resolution endpoint

```bash
curl -F "file=@input.png" \
  "http://localhost:4200/superres?out_format=jpeg&quality=92" \
  --output out.jpg
```

Supported formats:
  * png

  * jpeg


### 2. Super-resolution as a post-process in /generate

Add SR flags directly to the Stable Diffusion request.

Example JSON body:

```json
{
  "prompt": "a cinematic photograph of a futuristic city at sunset",
  "size": "512x512",
  "num_inference_steps": 4,
  "guidance_scale": 1.0,
  "seed": 12345678,
  "superres": true,
  "superres_format": "jpeg",
  "superres_quality": 92
}
```

The server will:
*  Run LCM Stable Diffusion

*  Pass the generated PNG to the SR worker

*  Return the upscaled image instead of the base output

### Response headers

When SR is enabled, responses include:

```code
X-SuperRes: 1
X-SR-Model: super-resolution-10.rknn
X-SR-Scale: 3
```

## Design notes

* The SR model is small (~512 KB), but memory usage is dominated by output buffers

* float32 intermediate arrays

* Input images are capped via SR_MAX_PIXELS to prevent runaway memory usage

* SR runs in dedicated worker threads, isolated from SD pipelines

* If SR is disabled, /generate behaves exactly like the original LCM server

## Why this exists

This server turns the original RKNN super-resolution demo into a production-safe service that works alongside Stable Diffusion, avoids RKNN thread-safety pitfalls, supports concurrent requests, and enables SR as a composable post-processing step.

It is intentionally conservative and explicit with optimizations for stability on embedded NPUs, not desktop GPUs.



# Stable Diffusion 1.5 Latent Consistency Model (LCM-SD) for RKNN2

Run the **Stable Diffusion 1.5 Latent Consistency Model (LCM-SD)** on **Rockchip RKNPU2 (RK3588)** using RKNN2.

This repository supports **command-line inference** and a **production-ready HTTP server** optimized specifically for **LCM-SD**.

---

## Performance (RK3588, single NPU core)

| Resolution | Text Encoder | U-Net (per step) | VAE Decoder |
|-----------:|-------------:|-----------------:|------------:|
| 384×384    | ~0.05s       | ~2.36s           | ~5.48s      |
| 512×512    | ~0.05s       | ~5.65s           | ~11–14s     |

> NOTE: VAE decode latency is a known RKNN limitation and is not caused by layout, server, or postprocessing overhead.

---

## LCM-SD Optimizations & Quirks (Specific to This Repo)

- Correct tensor layouts:
  - Text encoder: **NCHW**
  - U-Net: **NHWC**
  - VAE decoder: **NHWC**
- All RKNN runtime auto-conversion warnings eliminated
- One RKNN runtime context per worker (safe multi-context usage)
- Deterministic generation via explicit `numpy.RandomState(seed)`
- VAE decode slowness is a **known RKNN behavior** and unaffected by toolkit version

---

## Command-Line Usage (LCM-SD Only)

```bash
python ./run_rknn-lcm.py -i ./model -o ./images --num-inference-steps 4 -s 512x512 --prompt "Majestic mountain landscape with snow-capped peaks, autumn foliage in vibrant reds and oranges, a turquoise river winding through a valley, crisp and serene atmosphere, ultra-realistic style."
```

![image/png](https://cdn-uploads.huggingface.co/production/uploads/6319d0860d7478ae0069cd92/50jwBxv0Edf7x0WoHmpwi.png)

## LCM-SD HTTP Server

### Start the Server (Command Line)

```bash
export MODEL_ROOT=./model
export NUM_WORKERS=3
export PORT=4200

python lcm_server.py
```

The server listens on:

```bash
http://0.0.0.0:4200
```

## Server Endpoints (LCM-SD Only)

### POST /generate

Generate a PNG image using LCM-SD.

Request body (JSON):

```json
{
  "prompt": "a cinematic forest at sunrise",
  "size": "512x512",
  "num_inference_steps": 4,
  "guidance_scale": 1.0,
  "seed": 1234
}
```

Response:
  * HTTP 200
  * Content-Type: image/png
  * Binary PNG payload

### curl Example (LCM-SD Server Only)

```bash
curl -X POST http://localhost:4200/generate \
  -H "Content-Type: application/json" \
  -o output.png \
  -d '{
    "prompt": "a cinematic forest at sunrise",
    "size": "512x512",
    "num_inference_steps": 4,
    "guidance_scale": 1.0,
    "seed": 1234
  }'
```

## Docker Usage (LCM-SD Server)

### Build Image

```bash
docker build \
  -t rknn-lcm-sd .
```

### Run Container

```bash
docker run --rm -it \
  --device /dev/dri \
  --device /dev/rknpu \
  -v ./model:/models \
  -e MODEL_ROOT=/models \
  -e NUM_WORKERS=3 \
  -p 4200:4200 \
  rknn-lcm-sd
```

Additionally, a docker-compose.yml is provided.

## Model Conversion

### Install dependencies

```bash
pip install diffusers pillow numpy<2 rknn-toolkit2
```

### 1. Download the model

Download a Stable Diffusion 1.5 LCM model in ONNX format and place it in the `./model` directory.

```bash
huggingface-cli download TheyCallMeHex/LCM-Dreamshaper-V7-ONNX
cp -r -L ~/.cache/huggingface/hub/models--TheyCallMeHex--LCM-Dreamshaper-V7-ONNX/snapshots/4029a217f9cdc0437f395738d3ab686bb910ceea ./model
```

In theory, you could also achieve LCM inference by merging the LCM Lora into a regular Stable Diffusion 1.5 model and then converting it to ONNX format. However, I'm not sure how to do this. If anyone knows, please feel free to submit a PR.

### 2. Convert the model

```bash
# Convert the model, 384x384 resolution
python ./convert-onnx-to-rknn.py -m ./model -r 384x384 
```

Note that the higher the resolution, the larger the model and the longer the conversion time. It's not recommended to use very high resolutions.

## Known Issues

1. ~~As of now, models converted using the latest version of rknn-toolkit2 (version 2.2.0) still suffer from severe precision loss, even when using fp16 data type. As shown in the image, the top is the result of inference using the ONNX model, and the bottom is the result using the RKNN model. All parameters are the same. Moreover, the higher the resolution, the more severe the precision loss. This is a bug in rknn-toolkit2.~~ (Fixed in v2.3.0)

2. Actually, the model conversion script can select multiple resolutions (e.g., "384x384,256x256"), but this causes the model conversion to fail. This is a bug in rknn-toolkit2.

## References

- [TheyCallMeHex/LCM-Dreamshaper-V7-ONNX](https://huggingface.co/TheyCallMeHex/LCM-Dreamshaper-V7-ONNX)
- [Optimum's LatentConsistencyPipeline](https://github.com/huggingface/optimum/blob/main/optimum/pipelines/diffusers/pipeline_latent_consistency.py)
- [happyme531/RK3588-stable-diffusion-GPU](https://github.com/happyme531/RK3588-stable-diffusion-GPU)
