# To get CT-CLIP Model
Go to the CTRATE dataset on huggingface to download the model and place it at `./models` under the name `CT_CLIP.pt`

# multi2vec-clip-inference
The inference container for the clip module

## Documentation

Documentation for this module can be found [here](https://weaviate.io/developers/weaviate/current/retriever-vectorizer-modules/multi2vec-clip.html).

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
