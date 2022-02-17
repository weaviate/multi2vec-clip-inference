#!/usr/bin/env python3

import os
import sys
import logging
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer

text_model_name = os.getenv('TEXT_MODEL_NAME')
if text_model_name is None or text_model_name == "":
  logging.error("Fatal: TEXT_MODEL_NAME is required")
  sys.exit(1)

clip_model_name = os.getenv('CLIP_MODEL_NAME')
if clip_model_name is None or clip_model_name == "":
  logging.error("Fatal: CLIP_MODEL_NAME is required")
  sys.exit(1)

if clip_model_name.startswith('openai/'):
  if clip_model_name != text_model_name:
    logging.error(
      "For OpenAI models the 'CLIP_MODEL_NAME' and 'TEXT_MODEL_NAME' must be the same!"
    )
    sys.exit(1)
  logging.info(
    "Downloading OpenAI CLIP model {} from huggingface model hub".format(clip_model_name)
  )
  clip_model = CLIPModel.from_pretrained(clip_model_name)
  clip_model.save_pretrained('./models/openai_clip')
  processor = CLIPProcessor.from_pretrained(clip_model_name)
  processor.save_pretrained('./models/openai_clip_processor')

else:
  logging.info("Downloading text model {} from huggingface model hub".format(text_model_name))
  text_model = SentenceTransformer(text_model_name)
  text_model.save('./models/text')

  logging.info("Downloading img model {} from huggingface model hub".format(clip_model_name))
  clip_model = SentenceTransformer(clip_model_name)
  clip_model.save('./models/clip')
