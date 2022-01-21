import io
import base64
from PIL import Image
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer


class ClipInput(BaseModel):
  texts: list = []
  images: list = []


class ClipResult:
  textVectors: list = []
  imageVectors: list = []

  def __init__(self, textVectors, imageVectors):
    self.textVectors = textVectors
    self.imageVectors = imageVectors


class Clip:
  img_model: SentenceTransformer
  text_model: SentenceTransformer

  def __init__(self, cuda, cuda_core):
    device = 'cpu'
    if cuda:
        device=cuda_core
    self.img_model = SentenceTransformer('./models/clip', device=device)
    self.text_model = SentenceTransformer('./models/text', device=device)

  def vectorize(self, payload: ClipInput):
    try:
      textVectors = self.vectorizeTexts(payload.texts)
      imageVectors = self.vectorizeImages(payload.images)
      result = ClipResult(self.convertVectorArrays(textVectors), self.convertVectorArrays(imageVectors))
      return result
    except (RuntimeError, TypeError, NameError, Exception) as e:
      print('vectorize error:', e)
      raise e

  def convertVectorArrays(self, vectorResults):
    vectors = []
    if len(vectorResults) > 0:
      for vector in vectorResults:
        vectors.append(vector.tolist())

    return vectors

  def vectorizeImages(self, images):
    try:
      imageFiles = [self.parseImage(image) for image in images]
      return self.img_model.encode(imageFiles)
    except (RuntimeError, TypeError, NameError, Exception) as e:
      print('vectorize images error:', e)
      raise e

  def vectorizeTexts(self, texts):
    try:
      return self.text_model.encode(texts)
    except (RuntimeError, TypeError, NameError, Exception) as e:
      print('vectorize texts error:', e)
      raise e

  # parseImage decodes the base64 and parses the image bytes into a
  # PIL.Image. If the image is not in RGB mode, e.g. for PNGs using a palette,
  # it will be converted to RGB. This makes sure that they work with
  # SentenceTransformers/Huggingface Transformers which seems to require a (3,
  # height, width) tensor
  def parseImage(self, base64_encoded_image_string):
    image_bytes = base64.b64decode(base64_encoded_image_string)
    img = Image.open(io.BytesIO(image_bytes))

    if img.mode != 'RGB':
        img = img.convert('RGB')

    return img
