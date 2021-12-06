from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from PIL import Image
import requests
import base64, os
import uuid

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

  def __init__(self):
    self.img_model = SentenceTransformer('./models/clip')
    self.text_model = SentenceTransformer('./models/text')

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
      filepaths = [self.saveImage(str(uuid.uuid1()), image) for image in images]
      imageFiles = [self.loadImage(filepath) for filepath in filepaths]
      return self.img_model.encode(imageFiles)
    except (RuntimeError, TypeError, NameError, Exception) as e:
      print('vectorize images error:', e)
      raise e
    finally:
      for filepath in filepaths:
        self.removeFile(filepath)

  def vectorizeTexts(self, texts):
    try:
      return self.text_model.encode(texts)
    except (RuntimeError, TypeError, NameError, Exception) as e:
      print('vectorize texts error:', e)
      raise e

  def loadImage(self, url_or_path):
    if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
      return Image.open(requests.get(url_or_path, stream=True).raw)
    else:
      return Image.open(url_or_path)

  def saveImage(self, id: str, image: str):
    try:
      filepath = id
      file_content = base64.b64decode(image)
      with open(filepath, "wb") as f:
        f.write(file_content)
      return filepath
    except Exception as e:
      print(str(e))
      return ""

  def removeFile(self, filepath: str):
    if os.path.exists(filepath):
      os.remove(filepath)
