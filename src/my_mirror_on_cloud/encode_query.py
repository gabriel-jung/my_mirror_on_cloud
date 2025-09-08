import numpy as np
import loguru

from transformers import CLIPModel, CLIPTokenizer
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
    return model, tokenizer


def vectorize_query(query: str, model: CLIPModel=load_fashion_clip_model()[0], tokenizer: CLIPTokenizer=load_fashion_clip_model()[1])-> np.ndarray:
    """
    Vectorize the query using the Fashion CLIP model and tokenizer.
    input: query (str), CLIP model, CLIP tokenizer
    output: vectorized query (np.ndarray)
    """
    inputs = tokenizer([query], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)

    logger.debug("get text_feature")
    # Transformer en vecteur numpy
    vector_query = text_features[0].cpu().numpy()
    logger.debug(vector_query)
    return vector_query


if __name__ == "__main__":
    model, tokenizer = load_fashion_clip_model()
    query = "A casual red dress"
    vector_query = vectorize_query(query, model, tokenizer)
    logger.info(f"Query: {query}")
    logger.info(f"Vector Query: {vector_query}")