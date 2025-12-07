# multi2vec-clip-inference
The inference container for the clip module

## Documentation

Documentation for this module can be found [here](https://weaviate.io/developers/weaviate/model-providers/transformers/embeddings-multimodal).

## Build Docker container

```
LOCAL_REPO="multi2vec-clip" \
  TEXT_MODEL_NAME="sentence-transformers/clip-ViT-B-32-multilingual-v1" \
  CLIP_MODEL_NAME="clip-ViT-B-32" \
  ./cicd/build.sh

```

## Run tests

```
LOCAL_REPO="multi2vec-clip" ./cicd/test.sh
```

## NVIDIA Jetson devices

In order to run CLIP models on [NVIDIA Jetson device](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/) one needs to have [NVIDIA JetPack](https://developer.nvidia.com/embedded/jetpack) installed and configured.

This module only supports Jetson devices with JetPack 6 configured (JetPack 7 support coming soon).

### JetPack 6

In order to run a CLIP embedding model using Jetson GPU's one needs to install [requirements-nvidia-jetpack6.txt](requirements-nvidia-jetpack6.txt):

```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements-nvidia-jetpack6.txt
```

Use [./cicd/download_model.sh](./cicd/download_model.sh) script to download a model:

```bash
./cicd/download_model.sh
```

Start CLIP inference server:

```bash
ENABLE_CUDA=1 uvicorn app:app --host 0.0.0.0 --port 8000
```

Run smoke tests:

```bash
uv pip install -r requirements-test.txt
python3 smoke_test.py
```
