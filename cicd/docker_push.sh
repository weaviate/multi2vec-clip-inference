#!/usr/bin/env bash

set -eou pipefail

remote_repo=${REMOTE_REPO?Variable REMOTE_REPO is required}
model_tag_name=${MODEL_TAG_NAME?Variable MODEL_TAG_NAME is required}
docker_username=${DOCKER_USERNAME?Variable DOCKER_USERNAME is required}
docker_password=${DOCKER_PASSWORD?Variable DOCKER_PASSWORD is required}
clip_model_name=${CLIP_MODEL_NAME:-""}
text_model_name=${TEXT_MODEL_NAME:-""}
clip_model_type=${CLIP_MODEL_TYPE:-""}
open_clip_model_name=${OPEN_CLIP_MODEL_NAME:-""}
open_clip_pretrained=${OPEN_CLIP_PRETRAINED:-""}
siglip_model_name=${SIGLIP_MODEL_NAME:-""}
colpali_engine_model_name=${COLPALI_ENGINE_MODEL_NAME:-""}
git_tag=$GITHUB_REF_NAME

function main() {
  init
  echo "git ref type is $GITHUB_REF_TYPE"
  echo "git ref name is $GITHUB_REF_NAME"
  echo "git tag is $git_tag"
  push_tag
}

function init() {
  docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
  docker buildx create --use
  echo "$docker_password" | docker login -u "$docker_username" --password-stdin
}

function push_tag() {
  if [ ! -z "$git_tag" ] && [ "$GITHUB_REF_TYPE" == "tag" ]; then
    tag_git="$remote_repo:$model_tag_name-$git_tag"
    tag_latest="$remote_repo:$model_tag_name-latest"
    tag="$remote_repo:$model_tag_name"

    echo "Tag & Push $tag, $tag_latest, $tag_git"
    docker buildx build --platform=linux/arm64,linux/amd64 \
      --build-arg "TEXT_MODEL_NAME=$text_model_name" \
      --build-arg "CLIP_MODEL_NAME=$clip_model_name" \
      --build-arg "CLIP_MODEL_TYPE=$clip_model_type" \
      --build-arg "OPEN_CLIP_MODEL_NAME=$open_clip_model_name" \
      --build-arg "OPEN_CLIP_PRETRAINED=$open_clip_pretrained" \
      --build-arg "SIGLIP_MODEL_NAME=$siglip_model_name" \
      --build-arg "COLPALI_ENGINE_MODEL_NAME=$colpali_engine_model_name" \
      --push \
      --tag "$tag_git" \
      --tag "$tag_latest" \
      --tag "$tag" \
      .
  fi
}

main "${@}"
