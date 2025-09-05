"""Catalogue search module on Weaviate's vector database."""
import sys
print(sys.path)
from dotenv import load_dotenv
import loguru 
from time import perf_counter

import numpy as np


import weaviate
from weaviate.classes.init import Auth
import weaviate.classes as wvc
from weaviate.classes.query import MetadataQuery

from my_mirror_on_cloud.params import WEAVIATE_URL, WEAVIATE_KEY, MISTRAL_API_KEY
from my_mirror_on_cloud.encode_query import vectorize_query

logger = loguru.logger



def weaviate_connect() -> weaviate.Client:
    """
    Connect to Weaviate instance.
    output: Weaviate client
    """
    load_dotenv(override=True)
  
    # Open a Weaviate client
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=Auth.api_key(WEAVIATE_KEY),

        headers = {
            "X-Mistral-Api-Key": MISTRAL_API_KEY
        },
    )

    if client.is_ready():
        logger.info("Weaviate client is ready")
        return client
    else:
        logger.error("Weaviate client is not ready")
        return None



def connect_collection(client: weaviate.Client):
    """
    Connect to the Weaviate collection and load the CLIP model and tokenizer.
    input: Weaviate client
    output: Weaviate collection, CLIP model, CLIP tokenizer, CLIP processor 
    """
    tenues_collection = client.collections.use("ImageCatalogue")
    clothes_collection = client.collections.use("Vetements")

    return tenues_collection, clothes_collection



def search_by_vector(vector_query: np.ndarray, collection)-> weaviate.classes.QueryResult:
    """
    Search for look/items corresponding to the text query.
    input: vector of an image or text, Weaviate collection
    output: Weaviate query result
    """
    result = collection.query.near_vector(
        near_vector = vector_query,
        limit=3,
        include_vector=True,
        return_metadata=MetadataQuery(certainty=True)
    )

    for o in result.objects:
        logger.info(o.properties)
        # similarity
        logger.info(o.metadata.certainty)
        #logger.info(o.vector["default"])
        #logger.info(o.uuid)

    return result


def get_similar_text_to_vector(query: str, collection: str)-> weaviate.classes.QueryResult:
    """
    Get the most similar look/item to the text query.
    input: text query, Weaviate collection
    output: Weaviate query result
    """

    t1 = perf_counter()
    vector_query = vectorize_query(query)
    t2 = perf_counter()
    # Add a "if" condition to use the correct collection
    result = search_by_vector(vector_query, clothes_collection)
    t3 = perf_counter()

    logger.info(f"Query: {query}")
    logger.info(f"Vectorization Time: {t2 - t1:.4f} seconds")
    logger.info(f"Search Time: {t3 - t2:.4f} seconds")

    return result


def get_clothes_associated_to_look(look_uuid: str, collection: weaviate.classes.Collection)-> weaviate.classes.QueryResult:
    """
    Get the clothes associated to a look.
    input: look uuid, Weaviate collection
    output: Weaviate query result
    """
    result = collection.query.get(  
        # find property look_uuid = look_uuid       
        where={
            "path": ["tenue_uuid"],
            "operator": "Equal",
            "valueString": look_uuid
        },
        properties=["name", "description", "image_url", "clothes { name description image_url price link }"]
    )

    for o in result.objects:
        logger.info(o.properties)
        #logger.info(o.metadata.certainty)
        #logger.info(o.vector["default"])
        #logger.info(o.uuid)

    return result


if __name__ == "__main__":
    client = weaviate_connect()
    tenue_collection, clothes_collection = connect_collection(client)
    logger.info(clothes_collection.info())
    
    query = "A chic red dress"
    result = get_similar_text_to_vector(query, tenue_collection)
    logger.info(result)

    for o in result.objects:
        look_uuid = o.uuid
        logger.info(f"Look UUID: {look_uuid}")
        #result_clothes = get_clothes_associated_to_look(look_uuid, clothes_collection)
        #logger.info(result_clothes)