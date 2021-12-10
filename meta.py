from transformers import CLIPConfig
from transformers import AutoConfig

class Meta:
  clip_config: CLIPConfig

  def __init__(self, clip_model_path, text_model_path):
    self.clip_config = CLIPConfig.from_pretrained(clip_model_path + '/0_CLIPModel')
    try:
      # try as if it was a regular hf model
      self.text_config = AutoConfig.from_pretrained(text_model_path)
    except (RuntimeError, TypeError, NameError, Exception, EnvironmentError) as e:
      # no try as if it's a ST CLIP model
      self.text_config = AutoConfig.from_pretrained(text_model_path + '/0_CLIPModel')

  def get(self):
    return {
      'clip_model': self.clip_config.to_dict(),
      'text_model': self.text_config.to_dict(),
    }
