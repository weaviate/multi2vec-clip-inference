#!/usr/bin/env bash

set -e

local_repo=${LOCAL_REPO?Variable LOCAL_REPO is required}
text_model_name=${TEXT_MODEL_NAME}
clip_model_name=${CLIP_MODEL_NAME}
open_clip_model_name=${OPEN_CLIP_MODEL_NAME}
open_clip_pretrained=${OPEN_CLIP_PRETRAINED}

docker build \
  --build-arg "TEXT_MODEL_NAME=$text_model_name" \
  --build-arg "CLIP_MODEL_NAME=$clip_model_name" \
  --build-arg "OPEN_CLIP_MODEL_NAME=$open_clip_model_name" \
  --build-arg "OPEN_CLIP_PRETRAINED=$open_clip_pretrained" \
  -t "$local_repo" .
