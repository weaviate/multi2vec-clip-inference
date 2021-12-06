#!/usr/bin/env bash

set -e

local_repo=${LOCAL_REPO?Variable LOCAL_REPO is required}
text_model_name=${TEXT_MODEL_NAME?Variable TEXT_MODEL_NAME is required}
clip_model_name=${CLIP_MODEL_NAME?Variable CLIP_MODEL_NAME is required}

docker build --build-arg "TEXT_MODEL_NAME=$text_model_name"  --build-arg "CLIP_MODEL_NAME=$clip_model_name" -t "$local_repo" .
