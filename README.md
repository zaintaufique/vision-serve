# vision-serve

[![CI](https://github.com/zaintaufique/vision-serve/actions/workflows/ci.yml/badge.svg)](https://github.com/zaintaufique/vision-serve/actions/workflows/ci.yml)

A production inference platform for computer vision models. One repository, one shared serving core, and many independent applications: each application ships as its own container image containing only the code it needs.

The design goal is the one a real ML platform team faces: adding a second model to a service should not mean copying the service, and shipping the second model should not mean shipping the first one's code alongside it.

---

## Architecture

The system splits into two layers, and the dependency arrow points only one way.

```
  apps/imagenet_resnet18  ──┐
  apps/casting            ──┼──►  core
  apps/<future>           ──┘

  core  ──X──►  apps          core NEVER imports an application
```

**`core/`** is the platform. It knows how to serve HTTP, run an ONNX or PyTorch session, decode and preprocess an image, and shape a response. It knows nothing about castings, ImageNet, or any other problem domain. It is stable and rarely changes.

**`apps/`** are the product lines. Each one knows everything about a single problem: its dataset, labels, model, input size and normalization constants. Adding one changes nothing else.

The single seam is `core/registry.py`, which holds one dictionary of application names. That is the only file in `core` that ever names an application. Everything else in `core` operates purely through the contract below.

### The App contract

An application must declare five things. Core handles the rest.

```
App
├── name           str                  identifier, matches the folder
├── labels         list[str]            class names, index-aligned to logits
├── input_spec     InputSpec            crop size, mean, std, colour mode,
│                                       optional shorter-side resize
├── model_path     Path                 where the ONNX artifact lives
└── postprocess()  logits -> responses  logits to the JSON response shape

plus two build-time hooks the application owns:
├── load_torch_model()   supplies a torch module to the PyTorch backend
└── export()             converts that module to ONNX at image build time
```

The split that matters: **the code that applies a normalization lives in `core`, the values it applies live in the app.** `core/image.py` receives a mean and a standard deviation as arguments and never knows whose they are. This is what allows a new application to drop in without editing shared code.

---

## Why it is built this way

**One image per application, containing one application.** The runtime stage of the Dockerfile is parameterized by `ARG APP` and copies `core` plus exactly one application directory. The casting image contains no ImageNet labels, no ImageNet model, and no ImageNet code. There is still only one Dockerfile to maintain.

**Torch never reaches the shipped image.** The build has three stages: an exporter stage that installs PyTorch solely to produce the ONNX artifact, a builder stage that installs a clean torch-free runtime environment, and a runtime stage that copies only the finished venv and the finished model. Two stages would risk contamination, so there are three. The shipped image is 378 MB.

**The ONNX model is regenerated at build time, not copied from disk.** Model artifacts are gitignored, so a build that copied them from the working tree would succeed locally and fail on CI or a fresh clone. The exporter stage runs `python -m vision_serve.apps.${APP}.export` instead.

**Both backends share one preprocessing implementation.** Originally, the PyTorch path used torchvision's transforms while the ONNX path used a hand-written NumPy reproduction, and the two agreed to within a maximum logit difference of about 1.4e-05. They now both call `core/image.py`, so preprocessing is identical by construction rather than by reproduction, and the residual difference between backends (2.4e-07 on top-1 probability) is nothing but floating-point kernel variation.

**Backends are imported lazily.** `select_backend` imports torch only inside the PyTorch branch, so the ONNX-only production image, which has no torch installed, never attempts to import it.

---

## Applications

| Application | Task | Model | Status |
|---|---|---|---|
| `imagenet_resnet18` | 1000-class image classification | ResNet-18, ImageNet weights | Shipped |
| `casting` | Industrial defect detection on metal castings | Fine-tuned vision model | In progress |

### `imagenet_resnet18`

The reference application, and the first implementation of the App contract. A pretrained ResNet-18 exported to ONNX, serving top-k predictions over 1000 ImageNet classes. It exists partly to serve and partly to prove the contract holds: if the platform's oldest application does not fit the abstraction cleanly, the abstraction is wrong.

### `casting`

Binary defect classification on submersible pump impellers photographed on a real production line. This is the application the data pipeline work is built around.

**Dataset:** [Real-life industrial dataset of casting product](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product) (Kaggle). The download contains two sets: 7,348 images at 300x300 that arrive pre-split into train and test folders, and 1,300 images at 512x512 with no split.

**Why this pipeline discards the pre-split set.** The larger count cannot be explained by the smaller image size: you do not obtain more photographs by downscaling. The 300x300 set is the 512x512 originals inflated via augmentation, which raises the question of whether the augmentation was applied before or after the train-test split was drawn. If before, then rotated and flipped copies of the same physical casting appear on both sides of the split; the model is graded on parts it trained on, and the reported accuracy is fiction with no error to warn you. The construction of that split is undocumented and cannot be verified, so this pipeline does not use it. Splits are built from the 1,300 original images, and augmentation is applied only afterward, only to the training split, which makes that class of leakage structurally impossible.

**Known limitation, stated rather than hidden.** The source filenames carry a class and an opaque identifier, and nothing that identifies a physical part. If the same impeller was photographed more than once, there is no way to detect it, and duplicate captures may land on opposite sides of the split. This risk is inherited from the source data and cannot be eliminated, only acknowledged.

Full documentation of the dataset, its splits, and all cleaning decisions is in [`src/vision_serve/apps/casting/DATASET.md`](src/vision_serve/apps/casting/DATASET.md).

---

## Quickstart

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/zaintaufique/vision-serve.git
cd vision-serve

# install the runtime plus the export toolchain
uv sync --extra export --extra onnx

# produce the ONNX artifact for an application
uv run python -m vision_serve.apps.imagenet_resnet18.export

# serve it
VISION_SERVE_APP=imagenet_resnet18 \
VISION_SERVE_BACKEND=onnx \
  uv run uvicorn vision_serve.core.server:app --port 8000
```

Then:

```bash
curl -s -F "file=@dog.jpg" "http://localhost:8000/predict?top_k=3"
curl -s "http://localhost:8000/healthz"
```

### Configuration

| Variable | Values | Default |
|---|---|---|
| `VISION_SERVE_APP` | any name in `core/registry.py` | `imagenet_resnet18` |
| `VISION_SERVE_BACKEND` | `onnx`, `pytorch` | `pytorch` |

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/healthz` | Liveness probe. Reports the active application and backend. |
| `POST` | `/predict` | Classify an uploaded image. Query parameter `top_k` defaults to 5. |

---

## Docker

One Dockerfile builds every application. The application is selected at build time and the runtime stage copies only that one.

```bash
docker build --build-arg APP=imagenet_resnet18 -t vision-serve:imagenet .
docker build --build-arg APP=casting           -t vision-serve:casting  .

docker run -p 8000:8000 vision-serve:imagenet
```

`BACKEND` defaults to `onnx`, which is the intended production path. The PyTorch backend exists for local comparison and for exporting.

---

## Deploying to AWS

The service is deployed to **ECS Express Mode** in `eu-central-1`, fronted by an application load balancer. The steps are the same for any application, with the image tag being the only thing that changes.

```
1. BUILD     docker build --build-arg APP=<app> -t vision-serve:<app> .

2. PUSH      authenticate to ECR, tag the image against the ECR
             repository URI, and push it

3. DEPLOY    create an ECS Express service from the pushed image:
                port          8000
                health check  /healthz
                memory        1 GB   (512 MB out-of-memories on model load)
                tasks         1

4. VERIFY    POST an image to the public HTTPS endpoint the service
             returns, and confirm the prediction matches local output

5. TEAR DOWN delete the service
```

**No service is currently running.** The load balancer bills hourly, whether or not it serves traffic, so the service is deleted after every session. The image stays in ECR, where storage costs pennies, which means a redeploy skips the build entirely.

**One thing worth knowing:** on a freshly created AWS account, the first deployment can fail with an `AccessDenied` error even when the IAM policy is correct. IAM changes propagate asynchronously. Waiting briefly and redeploying to the existing role resolves it.

---

## Adding a new application

The abstraction is only real if this list is short. It is five steps, and none of them touch existing applications.

```
1. mkdir src/vision_serve/apps/<name>/

2. app.py       declare the contract:
                  NAME, labels, input_spec, model_path, postprocess()
                  load_torch_model()
                  APP = <YourApp>()

3. export.py    convert the torch module to ONNX at MODEL_PATH

4. registry.py  add "<name>" to APP_NAMES        ← the one line in core

5. tests/apps/<name>/   test the contract
```

Then build it:

```bash
docker build --build-arg APP=<name> -t vision-serve:<name> .
```

Nothing in `core/` changes except that one line in `APP_NAMES`. If a new application cannot be expressed through the contract, that is a signal the contract is missing something and should be extended deliberately, rather than a signal to special-case it inside `core`.

---

## Project layout

```
vision-serve/
├── Dockerfile                      one file, parameterized by ARG APP
├── pyproject.toml                  optional-dependency groups per backend
├── .github/workflows/ci.yml        ruff lint, ruff format, pytest
│
├── src/vision_serve/
│   ├── core/                       the platform: knows no application
│   │   ├── interface.py            the App contract and InputSpec
│   │   ├── registry.py             the only file in core naming apps
│   │   ├── server.py               FastAPI factory, /healthz, /predict
│   │   ├── runtime.py              OnnxBackend, TorchBackend
│   │   ├── image.py                generic decode, resize, normalize
│   │   └── config.py               environment-driven settings
│   │
│   └── apps/
│       ├── imagenet_resnet18/
│       │   ├── app.py              the contract, implemented
│       │   ├── export.py           torch to ONNX
│       │   └── labels.txt          1000 ImageNet class names
│       │
│       └── casting/
│           ├── app.py
│           ├── prepare_data.py     the reproducible data pipeline
│           ├── DATASET.md          the datasheet
│           └── labels.txt
│
├── tests/
│   ├── core/                       platform tests
│   └── apps/                       per-application tests
│
├── data/                           gitignored
│   ├── raw/                        immutable source images, never modified
│   └── manifests/                  split assignments, versioned in git
│
└── artifacts/                      gitignored: ONNX model files
```

**On `data/` and the manifest.** Splits are never materialized as copied folders of images. `data/raw/` is immutable and is the single source of truth. The split is a decision, recorded as a text manifest listing each image's path, label, and assigned split. That makes it deterministic, reproducible from a seed, and reviewable: changing the split ratio or adding new images produces a diff you can read, rather than a directory tree you have to trust.

---

## Development

```bash
uv run pytest                  # tests
uv run ruff check .            # lint
uv run ruff format --check .   # formatting
```

CI runs all three on every pull request. Branch protection requires a green run, and merges are squashed.

---

## License

The code in this repository is available under the MIT License. The casting dataset is subject to its own license on Kaggle and is not redistributed here; the data pipeline downloads it directly.
