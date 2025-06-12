#!/usr/bin/env bash

set -eou pipefail

local_repo=${LOCAL_REPO?Variable LOCAL_REPO is required}
text_model_name=${TEXT_MODEL_NAME:-""}
clip_model_name=${CLIP_MODEL_NAME:-""}
clip_model_type=${CLIP_MODEL_TYPE:-""}
open_clip_model_name=${OPEN_CLIP_MODEL_NAME:-""}
open_clip_pretrained=${OPEN_CLIP_PRETRAINED:-""}
siglip_model_name=${SIGLIP_MODEL_NAME:-""}

docker build \
  --build-arg "TEXT_MODEL_NAME=$text_model_name" \
  --build-arg "CLIP_MODEL_NAME=$clip_model_name" \
  --build-arg "CLIP_MODEL_TYPE=$clip_model_type" \
  --build-arg "OPEN_CLIP_MODEL_NAME=$open_clip_model_name" \
  --build-arg "OPEN_CLIP_PRETRAINED=$open_clip_pretrained" \
  --build-arg "SIGLIP_MODEL_NAME=$siglip_model_name" \
  -t "$local_repo" .
