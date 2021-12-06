from transformers import CLIPConfig
from transformers import AutoConfig

class Meta:
  clip_config: CLIPConfig

  def __init__(self, clip_model_path, text_model_path):
    self.clip_config = CLIPConfig.from_pretrained(clip_model_path + '/0_CLIPModel')
    self.text_config = AutoConfig.from_pretrained(text_model_path)

  def get(self):
    return {
      'clip_model': self.clip_config.to_dict(),
      'text_model': self.text_config.to_dict(),
    }
