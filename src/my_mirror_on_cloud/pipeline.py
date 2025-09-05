from .resume_client_query import reformulation_query
from .encode_query import vectorize_query
from .catalogue_search import get_similar_text_to_vector, weaviate_connect, connect_collection

def init_model():
    client = weaviate_connect()
    tenues_collection, clothes_collection = connect_collection(client)
    return tenues_collection, clothes_collection


def search_recommended_outfit(query, collection):
    cleaned_query = reformulation_query(query)
    # Need to check: il faut potentiellement cleaned the cleaned_query
    recommended_objects = get_similar_text_to_vector(cleaned_query, collection)
    return recommended_objects

