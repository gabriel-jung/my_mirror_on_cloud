from transformers import MarianMTModel, MarianTokenizer
from langdetect import detect
import loguru
logger = loguru.logger

def init_language():
    """
    
    """
    model_name = "Helsinki-NLP/opus-mt-fr-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return model, tokenizer


def translate_to_en(text, model, tokenizer):
    lang = detect(text)
    if lang == "fr":
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        translated = model.generate(**inputs)
        return tokenizer.decode(translated[0], skip_special_tokens=True)
    else:
        return text  # si déjà anglais ou langue non supportée




if __name__ == "__main__":
    model, tokenizer = init_language()
    text = "Bonjour tout le monde"
    trad = translate_to_en(text, model, tokenizer)
    logger.debug(trad)

