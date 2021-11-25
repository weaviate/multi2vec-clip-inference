from transformers import CLIPConfig

class Meta:
  config: CLIPConfig

  def __init__(self, model_path):
    self.config = CLIPConfig.from_pretrained(model_path)

  def get(self):
    return {
      'model': self.config.to_dict()
    }
