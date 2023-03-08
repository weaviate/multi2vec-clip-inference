#!/usr/bin/env bash

set -e pipefail

# Docker push rules
# If not on main
# - nothing is pushed
# If on main and not PR
# - any commit is pushed as :<model>-<7-digit-hash> 
# If on tag (e.g. 1.0.0)
# - any commit is pushed as :<model>-<semver>
# - any commit is pushed as :<model>-latest
# - any commit is pushed as :<model>
git_hash=
pr=
remote_repo=${REMOTE_REPO?Variable REMOTE_REPO is required}
model_tag_name=${MODEL_TAG_NAME?Variable MODEL_TAG_NAME is required}
docker_username=${DOCKER_USERNAME?Variable DOCKER_USERNAME is required}
docker_password=${DOCKER_PASSWORD?Variable DOCKER_PASSWORD is required}
clip_model_name=${CLIP_MODEL_NAME}
text_model_name=${TEXT_MODEL_NAME}
open_clip_model_name=${OPEN_CLIP_MODEL_NAME}
open_clip_pretrained=${OPEN_CLIP_PRETRAINED}

function main() {
  init
  echo "git branch is $GIT_BRANCH"
  echo "git tag is $GIT_TAG"
  echo "pr is $pr"
  push_main
  push_tag
}

function init() {
  git_hash="$(git rev-parse HEAD | head -c 7)"
  pr=false
  if [ ! -z "$GIT_PULL_REQUEST" ]; then
    pr="$GIT_PULL_REQUEST"
  fi

  docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
  docker buildx create --use
  echo "$docker_password" | docker login -u "$docker_username" --password-stdin
}

# Note that some CI systems, such as travis, will not provide the branch, but
# the tag on a tag-push. So this method will not be called on a tag-run.
function push_main() {
  if [ "$GIT_BRANCH" == "main" ] && [ "$pr" == "false" ]; then
    # The ones that are always pushed

    tag="$remote_repo:$model_tag_name-$git_hash"
    docker buildx build --platform=linux/arm64,linux/amd64 \
      --build-arg "TEXT_MODEL_NAME=$text_model_name" \
      --build-arg "CLIP_MODEL_NAME=$clip_model_name" \
      --build-arg "OPEN_CLIP_MODEL_NAME=$open_clip_model_name" \
      --build-arg "OPEN_CLIP_PRETRAINED=$open_clip_pretrained" \
      --push \
      --tag "$tag" .
  fi
}

function push_tag() {
  if [ ! -z "$GIT_TAG" ]; then
    tag_git="$remote_repo:$model_tag_name-$GIT_TAG"
    tag_latest="$remote_repo:$model_tag_name-latest"
    tag="$remote_repo:$model_tag_name"

    echo "Tag & Push $tag, $tag_latest, $tag_git"
    docker buildx build --platform=linux/arm64,linux/amd64 \
      --build-arg "TEXT_MODEL_NAME=$text_model_name" \
      --build-arg "CLIP_MODEL_NAME=$clip_model_name" \
      --build-arg "OPEN_CLIP_MODEL_NAME=$open_clip_model_name" \
      --build-arg "OPEN_CLIP_PRETRAINED=$open_clip_pretrained" \
      --push \
      --tag "$tag_git" \
      --tag "$tag_latest" \
      --tag "$tag" \
      .
  fi
}

main "${@}"
