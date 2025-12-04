import io
import base64
from os import path
from abc import ABC, abstractmethod
from typing import Union
from PIL import Image
from pydantic import BaseModel
from transformers import CLIPProcessor, CLIPModel, SiglipModel, AutoProcessor
from colpali_engine.models import BiModernVBert, BiModernVBertProcessor
from sentence_transformers import SentenceTransformer
import open_clip
import torch
import json
import asyncio
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from torch.nn.functional import normalize


class ClipInput(BaseModel):
	texts: list = []
	images: list = []


class ClipResult:
	text_vectors: list = []
	image_vectors: list = []

	def __init__(self, text_vectors, image_vectors):
		self.text_vectors = text_vectors
		self.image_vectors = image_vectors


class ClipInferenceABS(ABC):
	"""
	Abstract class for Clip Inference models that should be inherited from.
	"""

	@abstractmethod
	def vectorize(self, payload: ClipInput) -> ClipResult:
		...


class ClipInferenceSentenceTransformers(ClipInferenceABS):
	img_model: SentenceTransformer
	text_model: SentenceTransformer
	lock: Lock

	def __init__(self, cuda, cuda_core, trust_remote_code: bool):
		self.lock = Lock()
		device = 'cpu'
		if cuda:
			device = cuda_core

		self.img_model = SentenceTransformer('./models/clip', device=device, trust_remote_code=trust_remote_code)
		self.text_model = SentenceTransformer('./models/text', device=device, trust_remote_code=trust_remote_code)

	def vectorize(self, payload: ClipInput) -> ClipResult:
		"""
		Vectorize data from Weaviate.

		Parameters
		----------
		payload : ClipInput
			Input to the Clip model.

		Returns
		-------
		ClipResult
			The result of the model for both images and text.
		"""

		text_vectors = []
		if payload.texts:
			try:
				self.lock.acquire()
				text_vectors = (
					self.text_model
					.encode(payload.texts, convert_to_tensor=True)
					.tolist()
				)
			finally:
				self.lock.release()
		
		image_vectors = []
		if payload.images:
			try:
				self.lock.acquire()
				image_files = [_parse_image(image) for image in payload.images]
				image_vectors = (
					self.img_model
					.encode(image_files, convert_to_tensor=True)
					.tolist()
				)
			finally:
				self.lock.release()

		return ClipResult(
			text_vectors=text_vectors,
			image_vectors=image_vectors,
		)


class ClipInferenceOpenAI:
	clip_model: CLIPModel
	processor: CLIPProcessor
	lock: Lock

	def __init__(self, cuda, cuda_core):
		self.lock = Lock()
		self.device = 'cpu'
		if cuda:
			self.device=cuda_core
		self.clip_model = CLIPModel.from_pretrained('./models/openai_clip').to(self.device)
		self.processor = CLIPProcessor.from_pretrained('./models/openai_clip_processor')

	def vectorize(self, payload: ClipInput) -> ClipResult:
		"""
		Vectorize data from Weaviate.

		Parameters
		----------
		payload : ClipInput
			Input to the Clip model.

		Returns
		-------
		ClipResult
			The result of the model for both images and text.
		"""

		text_vectors = []
		if payload.texts:
			try:
				self.lock.acquire()
				inputs = self.processor(
					text=payload.texts,
					return_tensors="pt",
					padding=True,
				).to(self.device)

				# Taken from the HuggingFace source code of the CLIPModel
				text_outputs = self.clip_model.text_model(**inputs)
				text_embeds = text_outputs[1]
				text_embeds = self.clip_model.text_projection(text_embeds)

				# normalized features
				text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
				text_vectors = text_embeds.tolist()
			finally:
				self.lock.release()


		image_vectors = []
		if payload.images:
			try:
				self.lock.acquire()
				image_files = [_parse_image(image) for image in payload.images]
				inputs = self.processor(
					images=image_files,
					return_tensors="pt",
					padding=True,
				).to(self.device)

				# Taken from the HuggingFace source code of the CLIPModel
				vision_outputs = self.clip_model.vision_model(**inputs)
				image_embeds = vision_outputs[1]
				image_embeds = self.clip_model.visual_projection(image_embeds)

				# normalized features
				image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
				image_vectors = image_embeds.tolist()
			finally:
				self.lock.release()

		return ClipResult(
			text_vectors=text_vectors,
			image_vectors=image_vectors,
		)


class ClipInferenceOpenCLIP:
	lock: Lock

	def __init__(self, cuda, cuda_core):
		self.lock = Lock()
		self.device = 'cpu'
		if cuda:
			self.device=cuda_core

		cache_dir = './models/openclip'
		with open(path.join(cache_dir, "config.json")) as user_file:
			config = json.load(user_file)

		model_name = config['model_name']
		pretrained = config['pretrained']

		model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, cache_dir=cache_dir, device=self.device)
		if cuda:
			model = model.to(device=self.device)

		self.clip_model = model
		self.preprocess = preprocess
		self.tokenizer = open_clip.get_tokenizer(model_name)

	def vectorize(self, payload: ClipInput) -> ClipResult:
		"""
		Vectorize data from Weaviate.

		Parameters
		----------
		payload : ClipInput
			Input to the Clip model.

		Returns
		-------
		ClipResult
			The result of the model for both images and text.
		"""

		text_vectors = []
		if payload.texts:
			try:
				self.lock.acquire()
				with torch.no_grad(), torch.cuda.amp.autocast():
					text = self.tokenizer(payload.texts).to(self.device)
					text_features = self.clip_model.encode_text(text).to(self.device)
					text_features /= text_features.norm(dim=-1, keepdim=True)
				text_vectors = text_features.tolist()
			finally:
				self.lock.release()

		image_vectors = []
		if payload.images:
			try:
				self.lock.acquire()
				image_files = [self.preprocess_image(image) for image in payload.images]
				image_vectors = [self.vectorize_image(image) for image in image_files]
			finally:
				self.lock.release()

		return ClipResult(
			text_vectors=text_vectors,
			image_vectors=image_vectors,
		)

	def preprocess_image(self, base64_encoded_image_string):
		image_bytes = base64.b64decode(base64_encoded_image_string)
		img = Image.open(io.BytesIO(image_bytes))
		return self.preprocess(img).unsqueeze(0).to(device=self.device)

	def vectorize_image(self, image):
		with torch.no_grad(), torch.cuda.amp.autocast():
			image_features = self.clip_model.encode_image(image).to(self.device)
			image_features /= image_features.norm(dim=-1, keepdim=True)

		return image_features.tolist()[0]


class ClipInferenceSigCLIP:
	lock: Lock

	def __init__(self, cuda, cuda_core, trust_remote_code: bool):
		self.lock = Lock()
		self.device = 'cpu'
		if cuda:
			self.device=cuda_core

		cache_dir = './models/siglip'
		with open('./models/model_name', 'r') as f:
			model_name = f.read()
			self.model_name = model_name

		self.model: SiglipModel = SiglipModel.from_pretrained(cache_dir).to(self.device)
		self.processor = AutoProcessor.from_pretrained(self.model_name, cache_dir=cache_dir, trust_remote_code=trust_remote_code, use_fast=True)

	def vectorize(self, payload: ClipInput) -> ClipResult:
		"""
		Vectorize data from Weaviate.

		Parameters
		----------
		payload : ClipInput
			Input to the Clip model.

		Returns
		-------
		ClipResult
			The result of the model for both images and text.
		"""

		text_vectors = []
		if payload.texts:
			with self.lock, torch.no_grad():
				inputs = self.processor(text=payload.texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
				text_vectors = self.model.get_text_features(**inputs).to(self.device).tolist()

		image_vectors = []
		if payload.images:
			with self.lock, torch.no_grad():
				image_files = [_parse_image(image) for image in payload.images]
				for img in image_files:
					inputs = self.processor(images=[img], return_tensors="pt").to(self.device)
					image_vectors.append(self.model.get_image_features(**inputs).to(self.device).tolist()[0])

		return ClipResult(
			text_vectors=text_vectors,
			image_vectors=image_vectors,
		)


class ClipInferenceColPaliEngine:
	lock: Lock

	def __init__(self, cuda, cuda_core):
		self.lock = Lock()
		self.device = 'cpu'
		if torch.backends.mps.is_available():
			self.device = 'mps'
		if cuda:
			self.device=cuda_core

		with open('./models/model_name', 'r') as f:
			model_name = f.read()
			self.model_name = model_name

		self.processor = BiModernVBertProcessor.from_pretrained(
						self.model_name, 
						cache_dir="./models/clip_engine_processor",
    )
		self.model = BiModernVBert.from_pretrained(
						self.model_name,
						cache_dir="./models/clip_engine_model",
						trust_remote_code=True,
						dtype="auto",
						device_map="auto",
		).to(self.device)

	def _get_embeddings(self, outputs):
		if isinstance(outputs, dict):
			embeddings = outputs["embeddings"]
		else:
			embeddings = outputs
		vectors = []
		for emb in embeddings:
			vectors.append(normalize(emb, dim=-1).to(self.device).tolist())
		return vectors

	def vectorize(self, payload: ClipInput) -> ClipResult:
		"""
		Vectorize data from Weaviate.

		Parameters
		----------
		payload : ClipInput
			Input to the Clip model.

		Returns
		-------
		ClipResult
			The result of the model for both images and text.
		"""

		text_vectors = []
		if payload.texts:
			with self.lock, torch.no_grad():
				inputs = self.processor.process_texts(payload.texts).to(self.device)
				outputs = self.model(**inputs).to(self.device)
				text_vectors = self._get_embeddings(outputs)

		image_vectors = []
		if payload.images:
			with self.lock, torch.no_grad():
				image_files = [_parse_image(image) for image in payload.images]
				inputs = self.processor.process_images(image_files).to(self.device)
				outputs = self.model(**inputs).to(self.device)
				image_vectors = self._get_embeddings(outputs)

		return ClipResult(
			text_vectors=text_vectors,
			image_vectors=image_vectors,
		)


class Clip:

	clip: Union[ClipInferenceOpenAI, ClipInferenceSentenceTransformers, ClipInferenceOpenCLIP]
	executor: ThreadPoolExecutor

	def __init__(self, cuda, cuda_core, trust_remote_code: bool):
		self.executor = ThreadPoolExecutor()

		if path.exists('./models/openai_clip'):
			self.clip = ClipInferenceOpenAI(cuda, cuda_core)
		elif path.exists('./models/openclip'):
			self.clip = ClipInferenceOpenCLIP(cuda, cuda_core)
		elif path.exists('./models/siglip'):
			self.clip = ClipInferenceSigCLIP(cuda, cuda_core, trust_remote_code)
		elif path.exists('./models/clip_engine_model'):
			self.clip = ClipInferenceColPaliEngine(cuda, cuda_core)
		else:
			self.clip = ClipInferenceSentenceTransformers(cuda, cuda_core, trust_remote_code)

	async def vectorize(self, payload: ClipInput):
		"""
		Vectorize data from Weaviate.

		Parameters
		----------
		payload : ClipInput
			Input to the Clip model.

		Returns
		-------
		ClipResult
			The result of the model for both images and text.
		"""

		return await asyncio.wrap_future(self.executor.submit(self.clip.vectorize, payload))


# _parse_image decodes the base64 and parses the image bytes into a
# PIL.Image. If the image is not in RGB mode, e.g. for PNGs using a palette,
# it will be converted to RGB. This makes sure that they work with
# SentenceTransformers/Huggingface Transformers which seems to require a (3,
# height, width) tensor
def _parse_image(base64_encoded_image_string):
	image_bytes = base64.b64decode(base64_encoded_image_string)
	img = Image.open(io.BytesIO(image_bytes))

	if img.mode != 'RGB':
		img = img.convert('RGB')
	return img

