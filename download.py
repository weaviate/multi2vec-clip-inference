#!/usr/bin/env python3

import os
import sys
import logging
from transformers import CLIPProcessor, CLIPModel, SiglipModel, AutoTokenizer, AutoProcessor
from sentence_transformers import SentenceTransformer
import open_clip
import json

logging.basicConfig(level=logging.INFO)

def save_trust_remote_code(trust_remote_code: bool):
  if not os.path.exists("./models"):
    os.makedirs("./models")
  with open(f"./models/trust_remote_code", "w") as f:
    f.write(f"{trust_remote_code}")

trust_remote_code = os.getenv("TRUST_REMOTE_CODE", False)
save_trust_remote_code(trust_remote_code)

siglip_model_name = os.getenv('SIGLIP_MODEL_NAME')
if siglip_model_name is not None and siglip_model_name != "":
  cache_dir = "./models/siglip" 
  logging.info(f"Downloading siglip model {siglip_model_name}")
  model: SiglipModel = SiglipModel.from_pretrained(siglip_model_name)
  model.save_pretrained(save_directory=cache_dir)
  AutoProcessor.from_pretrained(siglip_model_name, cache_dir=cache_dir, trust_remote_code=trust_remote_code)
  with open(f"./models/model_name", "w") as f:
    f.write(f"{siglip_model_name}")
  sys.exit(0)


open_clip_model_name = os.getenv('OPEN_CLIP_MODEL_NAME')
open_clip_pretrained = os.getenv('OPEN_CLIP_PRETRAINED')

if open_clip_model_name is not None and open_clip_model_name != "" and open_clip_pretrained is not None and open_clip_pretrained != "":
  def check_model_and_pretrained(model_name: str, pretrained: str):
    if (model_name, pretrained) in open_clip.list_pretrained():
        return
    logging.error("Fatal: Available pairs are:")
    for pair in open_clip.list_pretrained():
        logging.error(f"Fatal: model: {pair[0]} pretrained: {pair[1]}")
    logging.error(f"Fatal: Match not found for OPEN_CLIP model {model_name} with pretrained {pretrained} pair")
    sys.exit(1)

  logging.info(f"Checking if OPEN_CLIP model {open_clip_model_name} and pretrained {open_clip_pretrained} is a valid pair")
  check_model_and_pretrained(open_clip_model_name, open_clip_pretrained)

  cache_dir = './models/openclip'
  logging.info(f"Downloading OPEN_CLIP model {open_clip_model_name} with pretrained {open_clip_pretrained} to cache dir {cache_dir}")
  model, _, preprocess = open_clip.create_model_and_transforms(open_clip_model_name, pretrained=open_clip_pretrained, cache_dir=cache_dir)
  model_config = open_clip.get_model_config(open_clip_model_name)

  config = {
    "model_name" : open_clip_model_name,
    "pretrained" : open_clip_pretrained,
    "model_config": model_config,
    "cache_dir" : cache_dir
  }

  with open(os.path.join(cache_dir, "config.json"), 'w') as f:
        json.dump(config, f)

  logging.info(f"Successfully downloaded and validated model and pretrained")
  sys.exit(0)

clip_model_type = os.getenv('CLIP_MODEL_TYPE', 'sentencetransformer')

text_model_name = os.getenv('TEXT_MODEL_NAME')
if text_model_name is None or text_model_name == "":
  logging.error("Fatal: TEXT_MODEL_NAME is required")
  sys.exit(1)

clip_model_name = os.getenv('CLIP_MODEL_NAME')
if clip_model_name is None or clip_model_name == "":
  logging.error("Fatal: CLIP_MODEL_NAME is required")
  sys.exit(1)

if clip_model_name.startswith('openai/') or clip_model_type.lower() == "openai" or clip_model_name.startswith('facebook/metaclip'):
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
  text_model = SentenceTransformer(text_model_name, trust_remote_code=trust_remote_code)
  text_model.save('./models/text')

  logging.info("Downloading img model {} from huggingface model hub".format(clip_model_name))
  clip_model = SentenceTransformer(clip_model_name, trust_remote_code=trust_remote_code)
  clip_model.save('./models/clip')
