from dataclasses import dataclass
import numpy as np 
from typing import Any

from .catalogue_search import (
    algo_flow,
    connect_collection,
)
from .manage_language import init_language, translate_to_en
from .weaviate_manager import WeaviateManager
from .embedding_manager import create_embedder 


@dataclass
class AlgoParams:
    wm: any
    tenues_col: str
    vet_col: str
    cat_col: str
    fashion_clip_emb: any
    type_of_query: str

def init_model()-> np.array:
    wm = WeaviateManager()
    tenues_collection, clothes_collection, catalogue_collection = connect_collection()
    fashion_clip_emb = create_embedder()
    model_lang, tokenizer_lang = init_language()
    return wm, tenues_collection, clothes_collection, catalogue_collection,  model_lang, tokenizer_lang, fashion_clip_emb


def search_recommended_outfit(query: str, img_path: Any, init_model: np.array, type_of_query: str)-> np.array:
    params = AlgoParams(init_model[0], init_model[1], init_model[2], init_model[3], init_model[6], type_of_query)
    trad_query = translate_to_en(query, init_model[4], init_model[5])
    #cleaned_query = reformulation_query(trad_query)
    cleaned_query = [trad_query]
    recommended_objects = algo_flow(cleaned_query, img_path, params)
    return recommended_objects
