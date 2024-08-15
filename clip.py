import io
import base64
from os import path
from abc import ABC, abstractmethod
from typing import Union
from PIL import Image
from pydantic import BaseModel
from transformers import CLIPProcessor, CLIPModel, BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
import open_clip
import torch
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import numpy as np
from ct_clip import CTCLIP
from transformer_maskgit import CTViT


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

	def __init__(self, cuda, cuda_core):
		self.lock = Lock()
		device = 'cpu'
		if cuda:
			device = cuda_core

		self.img_model = SentenceTransformer('./models/clip', device=device)
		self.text_model = SentenceTransformer('./models/text', device=device)

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

class CTClip:
	lock: Lock

	def __init__(self, cuda, cuda_core):
		self.lock = Lock()
		self.device = 'cpu'
		if cuda:
			self.device=cuda_core

		tokenizer = BertTokenizer.from_pretrained( 'microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True )

		text_encoder = BertModel.from_pretrained( "microsoft/BiomedVLP-CXR-BERT-specialized" )
		text_encoder.resize_token_embeddings( len( tokenizer ) )

		image_encoder = CTViT(
			dim = 512,
			codebook_size = 8192,
			image_size = 480,
			patch_size = 30,
			temporal_patch_size = 15,
			spatial_depth = 4,
			temporal_depth = 4,
			dim_head = 32,
			heads = 8
		)

		model = CTCLIP(
			image_encoder = image_encoder,
			text_encoder = text_encoder,
			dim_image = 2097152,
			dim_text = 768,
			dim_latent = 512,
			extra_latent_projection = False,		 # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
			use_mlm=False,
			downsample_image_embeds = False,
			use_all_token_embeds = False

		)

		model.load("./models/CT_CLIP.pt")
		if cuda:
			model = model.to(device=self.device)

		self.clip_model = model
		self.tokenizer = tokenizer 

	def preprocess( self, img_data ):
		img_data= np.transpose(img_data, (1, 2, 0)) 
		img_data = img_data*1000 
		hu_min, hu_max = -1000, 200 
		img_data = np.clip(img_data, hu_min, hu_max) 
 
		img_data = (((img_data+400 ) / 600)).astype(np.float32) 
		slices=[] 
 
		tensor = torch.tensor(img_data) 
		# Get the dimensions of the input tensor 
		target_shape = (480,480,240) 
		# Extract dimensions 
		h, w, d = tensor.shape 
 
		# Calculate cropping/padding values for height, width, and depth 
		dh, dw, dd = target_shape 
		h_start = max((h - dh) // 2, 0) 
		h_end = min(h_start + dh, h) 
		w_start = max((w - dw) // 2, 0) 
		w_end = min(w_start + dw, w) 
		d_start = max((d - dd) // 2, 0) 
		d_end = min(d_start + dd, d) 
 
		# Crop or pad the tensor 
		tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end] 
 
		pad_h_before = (dh - tensor.size(0)) // 2 
		pad_h_after = dh - tensor.size(0) - pad_h_before 
 
		pad_w_before = (dw - tensor.size(1)) // 2 
		pad_w_after = dw - tensor.size(1) - pad_w_before 
 
		pad_d_before = (dd - tensor.size(2)) // 2 
		pad_d_after = dd - tensor.size(2) - pad_d_before 
 
		tensor = torch.nn.functional.pad(tensor, (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after), value=-1) 
 
 
		tensor = tensor.permute(2, 0, 1) 

		tensor = tensor.unsqueeze( 0 )

		return tensor

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

		while len( payload.images ) < len( payload.texts ):
			payload.images.append( None )
		while len( payload.texts ) < len( payload.texts ):
			payload.texts.append( "" )

		text_vectors = []
		image_vectors = []
		try:
			self.lock.acquire()
			for batch in zip( payload.texts, payload.images ):
				with torch.no_grad(), torch.cuda.amp.autocast():
					text = batch[ 0 ].replace('"', '')
					text = text.replace('\'', '')
					text = text.replace('(', '')
					text = text.replace(')', '')
					text = self.tokenizer( text, return_tensors="pt", padding="max_length", truncation=True, max_length=512 ).to( self.device )
					image = self.preprocess_image( batch[ 1 ] )
					print( image.shape )

					out = self.clip_model( text, image, return_latents = True, device=self.device )

				text_vectors.append( out[ 0 ] )
				image_vectors.append( out[ 1 ] )
		finally:
			self.lock.release()

		return ClipResult(
			text_vectors=text_vectors,
			image_vectors=image_vectors,
		)

	def preprocess_image(self, base64_encoded_image_string):
		if base64_encoded_image_string:
			image_bytes = base64.b64decode(base64_encoded_image_string)
			img = np.load(io.BytesIO(image_bytes))
		else:
			img = np.zeros( ( 480, 480, 240 ) )
		return self.preprocess( img ).unsqueeze(0).to(device=self.device)

class Clip:

	clip: Union[ClipInferenceOpenAI, ClipInferenceSentenceTransformers, ClipInferenceOpenCLIP]
	executor: ThreadPoolExecutor

	def __init__(self, cuda, cuda_core):
		self.executor = ThreadPoolExecutor()

		if path.exists('./models/openai_clip'):
			self.clip = ClipInferenceOpenAI(cuda, cuda_core)
		elif path.exists('./models/openclip'):
			self.clip = ClipInferenceOpenCLIP(cuda, cuda_core)
		elif path.exists( "./models/CT_CLIP.pt" ):
			self.clip = CTClip( cuda, cuda_core )
		else:
			self.clip = ClipInferenceSentenceTransformers(cuda, cuda_core)

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

