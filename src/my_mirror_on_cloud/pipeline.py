from .resume_client_query import reformulation_query
from .encode_query import vectorize_query
from .catalogue_search import get_similar_text_to_vector, weaviate_connect, connect_collection
from .manage_language import init_language, translate_to_en

def init_model():
    # Weaviate
    client = weaviate_connect()
    tenues_collection, clothes_collection = connect_collection(client)

    # FashionClip

    # description de Gabriel

    # language
    model_lang, tokenizer_lang = init_language()

    return tenues_collection, clothes_collection, model_lang, tokenizer_lang




def search_recommended_outfit(query, collection, model_lang, tokenizer_lang):
    trad_query = translate_to_en(query, model_lang, tokenizer_lang)
    cleaned_query = reformulation_query(trad_query)
    # Need to check: il faut potentiellement cleaned the cleaned_query
    recommended_objects = get_similar_text_to_vector(cleaned_query, collection)
    return recommended_objects

