#!/usr/bin/env python3

from sentence_transformers import SentenceTransformer
import os
import sys

text_model_name = os.getenv('TEXT_MODEL_NAME')
if text_model_name is None or text_model_name == "":
  print("Fatal: TEXT_MODEL_NAME is required")
  sys.exit(1)

clip_model_name = os.getenv('CLIP_MODEL_NAME')
if clip_model_name is None or clip_model_name == "":
  print("Fatal: CLIP_MODEL_NAME is required")
  sys.exit(1)

print("Downloading text model {} from huggingface model hub".format(text_model_name))
text_model = SentenceTransformer(text_model_name)
text_model.save('./models/text')

print("Downloading img model {} from huggingface model hub".format(clip_model_name))
clip_model = SentenceTransformer(clip_model_name)
clip_model.save('./models/clip')

