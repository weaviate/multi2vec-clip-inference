import os
from logging import getLogger
from fastapi import FastAPI, Response, status
from contextlib import asynccontextmanager
from clip import Clip, ClipInput
from meta import Meta



clip : Clip
meta_config : Meta
logger = getLogger('uvicorn')


@asynccontextmanager
async def lifespan(app: FastAPI):
	global clip
	global meta_config

	def get_trust_remote_code() -> bool:
		if os.path.exists("./models/trust_remote_code"):
			with open("./models/trust_remote_code", "r") as f:
				trust_remote_code = f.read()
				return trust_remote_code == "true"
		return os.getenv("TRUST_REMOTE_CODE", False)

	trust_remote_code = get_trust_remote_code()

	cuda_env = os.getenv("ENABLE_CUDA")
	cuda_support=False
	cuda_core=""

	if cuda_env is not None and cuda_env == "true" or cuda_env == "1":
		cuda_support=True
		cuda_core = os.getenv("CUDA_CORE")
		if cuda_core is None or cuda_core == "":
			cuda_core = "cuda:0"
		logger.info(f"CUDA_CORE set to {cuda_core}")
	else:
		logger.info("Running on CPU")

	clip = Clip(cuda_support, cuda_core, trust_remote_code)
	meta_config = Meta()
	logger.info("Model initialization complete")
	yield

app = FastAPI(lifespan=lifespan)

@app.get("/.well-known/live", response_class=Response)
@app.get("/.well-known/ready", response_class=Response)
async def live_and_ready(response: Response):
	response.status_code = status.HTTP_204_NO_CONTENT


@app.get("/meta")
async def meta():
	return await meta_config.get()


@app.post("/vectorize")
async def read_item(payload: ClipInput, response: Response):
	try:
		result = await clip.vectorize(payload)
		return {
			"textVectors": result.text_vectors,
			"imageVectors": result.image_vectors
		}
	except Exception as e:
		logger.exception(
            'Something went wrong while vectorizing data.'
        )
		response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
		return {"error": str(e)}
