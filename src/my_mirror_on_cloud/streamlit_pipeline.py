from .resume_client_query import reformulation_query
from .catalogue_search import (
    algo_flow,
    connect_collection,
)
from .manage_language import init_language, translate_to_en
from .weaviate_manager import WeaviateManager
from .embedding_manager import create_embedder 


def init_model():
    wm = WeaviateManager()
    tenues_collection, clothes_collection, catalogue_collection = connect_collection()
    fashion_clip_emb = create_embedder()
    model_lang, tokenizer_lang = init_language()
    return wm, tenues_collection, clothes_collection, catalogue_collection,  model_lang, tokenizer_lang, fashion_clip_emb


def search_recommended_outfit(query, init_model):
    trad_query = translate_to_en(query, init_model[4], init_model[5])
    cleaned_query = reformulation_query(trad_query)
    recommended_objects = algo_flow(init_model[0], cleaned_query, init_model[1], init_model[2], init_model[3], init_model[6])
    return recommended_objects
