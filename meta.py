import json
from os import path
from transformers import CLIPConfig
from transformers import AutoConfig

class Meta:

  def __init__(self):
    if path.exists('./models/openai_clip'):
      # OpenAI CLIP Models
      self._config = CLIPConfig.from_pretrained('./models/openai_clip').to_dict()
    elif path.exists('./models/openclip'):
      # OpenCLIP Models
      with open(path.join('./models/openclip', "config.json")) as config_file:
        self._config = json.load(config_file)
    else:
      # Non OpenAI CLIP Models
      self._config = {
        'clip_model':  CLIPConfig.from_pretrained('./models').to_dict(),
      }
      try:
        # try as if it was a regular hf model
        self._config['text_model'] = AutoConfig.from_pretrained('./models/text').to_dict()
      except (RuntimeError, TypeError, NameError, Exception, EnvironmentError):
        # now try as if it's a ST CLIP model
        self._config['text_model'] = AutoConfig.from_pretrained('./models/text/0_CLIPModel').to_dict()
    
    self._config = json.loads(json.dumps(self._config, default=str))

  async def get(self):
    return self._config
