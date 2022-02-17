from os import path
from transformers import CLIPConfig
from transformers import AutoConfig

class Meta:
  clip_config: CLIPConfig

  def __init__(self):
    if path.exists('./models/openai_clip'):
      # OpenAI CLIP Models
      self._config = CLIPConfig.from_pretrained('./models/openai_clip').to_dict()
    else:
      # Non OpenAI CLIP Models
      self._config = {
        'clip_model':  CLIPConfig.from_pretrained('./models/clip/0_CLIPModel'),
      }
      try:
        # try as if it was a regular hf model
        self._config['text_model'] = AutoConfig.from_pretrained('./models/text')
      except (RuntimeError, TypeError, NameError, Exception, EnvironmentError) as e:
        # now try as if it's a ST CLIP model
        self._config['text_model'] = AutoConfig.from_pretrained('./models/text/0_CLIPModel')

  def get(self):
    return self._config
