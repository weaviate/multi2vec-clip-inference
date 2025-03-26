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
