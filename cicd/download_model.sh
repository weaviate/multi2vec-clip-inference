#!/usr/bin/env bash

set -eou pipefail

echo "=================================================="
echo "     multi2vec-clip Model Downloader"
echo "=================================================="
echo
echo "Choose a model to download:"
echo " 1) sentence-transformers/clip-ViT-B-32"
echo " 2) sentence-transformers/clip-ViT-B-32-multilingual-v1"
echo " 3) openai/clip-vit-base-patch16"
echo " 4) OpenCLIP ViT-B-16 pretrained laion2b_s34b_b88k"
echo " 5) OpenCLIP ViT-B-32-quickgelu pretrained laion400m_e32"
echo " 6) OpenCLIP xlm-roberta-base-ViT-B-32 pretrained laion5b_s13b_b90k"
echo " 7) SigLIP google/siglip-so400m-patch16-256-i18n"
echo " 8) SigLIP google/siglip2-so400m-patch16-384"
echo " 9) SigLIP google/siglip2-so400m-patch16-512"
echo "10) MetaCLIP facebook/metaclip-2-worldwide-b32-384"
echo "11) ModernVBERT/modernvbert-embed (ColPali engine)"
echo
read -p "Enter your choice (1-11): " choice

MODEL_NAME=""
PRETRAINED=""

case $choice in
  1)  MODEL_NAME="sentence-transformers/clip-ViT-B-32" ;;
  2)  MODEL_NAME="sentence-transformers/clip-ViT-B-32-multilingual-v1" ;;
  3)  MODEL_NAME="openai/clip-vit-base-patch16" ;;
  4)  MODEL_NAME="ViT-B-16"; PRETRAINED="laion2b_s34b_b88k" ;;
  5)  MODEL_NAME="ViT-B-32-quickgelu"; PRETRAINED="laion400m_e32" ;;
  6)  MODEL_NAME="xlm-roberta-base-ViT-B-32"; PRETRAINED="laion5b_s13b_b90k" ;;
  7)  MODEL_NAME="google/siglip-so400m-patch16-256-i18n" ;;
  8)  MODEL_NAME="google/siglip2-so400m-patch16-384" ;;
  9)  MODEL_NAME="google/siglip2-so400m-patch16-512" ;;
  10) MODEL_NAME="facebook/metaclip-2-worldwide-b32-384" ;;
  11) MODEL_NAME="ModernVBERT/modernvbert-embed" ;;
  *)
    echo "Invalid choice!"
    exit 1
    ;;
esac

if [[ $PRETRAINED != "" ]]; then
  echo "Starting download of $MODEL_NAME pretrained $PRETRAINED..."
else
  echo "Starting download of $MODEL_NAME..."
fi

rm -fr ./models

echo "choice: $choice"

if [[ $choice -eq 1 ]] || [[ $choice -eq 2 ]] || [[ $choice -eq 3 ]] || [[ $choice -eq 10 ]]; then
  CLIP_MODEL_NAME=$MODEL_NAME TEXT_MODEL_NAME=$MODEL_NAME ./download.py
elif [[ $choice -eq 4 ]] || [[ $choice -eq 5 ]] || [[ $choice -eq 6 ]]; then
  OPEN_CLIP_MODEL_NAME=$MODEL_NAME OPEN_CLIP_PRETRAINED=$PRETRAINED ./download.py
elif [[ $choice -eq 7 ]] || [[ $choice -eq 8 ]] || [[ $choice -eq 9 ]]; then
  SIGLIP_MODEL_NAME=$MODEL_NAME ./download.py
elif [[ $choice -eq 11 ]]; then
  COLPALI_ENGINE_MODEL_NAME=$MODEL_NAME ./download.py
fi

echo
echo "=================================================="
echo "Download complete!"
echo "=================================================="

exit 0
