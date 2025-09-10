from .resume_client_query import reformulation_query
from .catalogue_search import (
    algo_flow,
    connect_collection,
)
from .manage_language import init_language, translate_to_en


def init_model():
    # Weaviate
    #client = weaviate_connect()
    tenues_collection, clothes_collection, catalogue_collection = connect_collection()

    # FashionClip

    # description de Gabriel

    # language
    model_lang, tokenizer_lang = init_language()

    return tenues_collection, clothes_collection, catalogue_collection,  model_lang, tokenizer_lang




def search_recommended_outfit(query, tenues_collection, clothes_collection, catalogue_collection, model_lang, tokenizer_lang):
    trad_query = translate_to_en(query, model_lang, tokenizer_lang)
    cleaned_query = reformulation_query(trad_query)
  
    #recommended_objects = get_similar_text_to_vector(cleaned_query, collection)
    recommended_objects = algo_flow(cleaned_query, tenues_collection, clothes_collection, catalogue_collection)
    return recommended_objects
