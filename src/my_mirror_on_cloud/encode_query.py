import numpy as np
import loguru
from PIL import Image

from transformers import CLIPModel, CLIPTokenizer, CLIPProcessor
import torch

logger = loguru.logger

def load_fashion_clip_model(model_name: str = "patrickjohncyh/fashion-clip") -> tuple[CLIPModel, CLIPTokenizer]:
    """
    Load the Fashion CLIP model and tokenizer.
    input: model name (str)
    output: CLIP model, CLIP tokenizer
    """
    model = CLIPModel.from_pretrained(model_name)
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, tokenizer, processor


def vectorize_query(image, model: CLIPModel=load_fashion_clip_model()[0], processor: CLIPProcessor=load_fashion_clip_model()[2])-> np.ndarray:
    """
    Vectorize the query using the Fashion CLIP model and tokenizer.
    input: query (str), CLIP model, CLIP tokenizer
    output: vectorized query (np.ndarray)
    """
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)

    # Transformer en vecteur numpy
    image_vector = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
   
    return image_vector.squeeze().tolist()


if __name__ == "__main__":
    model, tokenizer = load_fashion_clip_model()
    query = "A casual red dress"
    vector_query = vectorize_query(query, model, tokenizer)
    logger.info(f"Query: {query}")
    logger.info(f"Vector Query: {vector_query}")