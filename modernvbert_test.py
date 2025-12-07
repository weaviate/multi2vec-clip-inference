import torch
from colpali_engine.models import BiModernVBert, BiModernVBertProcessor
from PIL import Image
from huggingface_hub import hf_hub_download
from torch.nn.functional import normalize

def get_device():
    if torch.cuda.is_available():
        return "cuda:0"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

model_id = "ModernVBERT/modernvbert-embed"

processor = BiModernVBertProcessor.from_pretrained(model_id)
model = BiModernVBert.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
)

image = Image.open(hf_hub_download("HuggingFaceTB/SmolVLM", "example_images/rococo.jpg", repo_type="space"))
text = "This is a text"
text2 = "This is a text2"

# Prepare inputs
text_inputs = processor.process_texts([text, text2]).to(get_device())
image_inputs = processor.process_images([image]).to(get_device())

text_outputs = model(**text_inputs).to(get_device())
image_outputs = model(**image_inputs).to(get_device())

def get_embeddings(outputs):
    with torch.no_grad():
        if isinstance(outputs, dict):
            embeddings = outputs["embeddings"]
        else:
            embeddings = outputs
    emb = normalize(embeddings[0], dim=-1).to(get_device()).tolist()
    return emb

text_emb = get_embeddings(text_outputs)
image_emb = get_embeddings(image_outputs)

print(f"text_emb: {len(text_emb)} first3: {text_emb[:3]}")
print(f"image_emb: {len(image_emb)} first3: {image_emb[:3]}")