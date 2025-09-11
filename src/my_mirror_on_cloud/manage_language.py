# Solution 1: langdetect
from transformers import MarianMTModel, MarianTokenizer
# Solution 2: CroissantLLM
#import torch
#from transformers import AutoModelForCausalLM, AutoTokenizer
import langid

import loguru
logger = loguru.logger

def init_language():
    # Solution 1
    model_name = "Helsinki-NLP/opus-mt-fr-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    # Solution 2
    # model_path = "./models/croissant-llm"
    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    # model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    return model, tokenizer


def translate_to_en(text, model, tokenizer):
    lang = langid.classify(text)[0]
    if lang == "fr":
        inputs = tokenizer(text, return_tensors="pt", padding=True).to(model.device)
        translated = model.generate(**inputs, max_new_tokens=100)
        return tokenizer.decode(translated[0], skip_special_tokens=True)
    else:
        return text  


if __name__ == "__main__":
    model, tokenizer = init_language()
    text = "Bonjour tout le monde"
    trad = translate_to_en(text, model, tokenizer)
    logger.debug(trad)

