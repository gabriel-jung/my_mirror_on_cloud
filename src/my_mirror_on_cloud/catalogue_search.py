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

from .params import WEAVIATE_URL, WEAVIATE_KEY, MISTRAL_API_KEY
from .encode_query import vectorize_query

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



def search_by_vector(vector_query: np.ndarray, collection):
    """
    Search for look/items corresponding to the text query.
    input: vector of an image or text, Weaviate collection
    output: Weaviate query result
    """
    logger.debug(collection)
    logger.debug(vector_query)
    result = collection.query.near_vector(
        near_vector = vector_query,
        limit=3,
        include_vector=True,
        return_metadata=MetadataQuery(certainty=True)
    )
    logger.debug(result)

    for o in result.objects:
        logger.info(o.properties)
        # similarity
        logger.info(o.metadata.certainty)
        #logger.info(o.vector["default"])
        #logger.info(o.uuid)

    return result


def get_similar_text_to_vector(query: str, collection):
    """
    Get the most similar look/item to the text query.
    input: text query, Weaviate collection
    output: Weaviate query result
    """

    t1 = perf_counter()
    logger.debug("start vectorize_query")
    vector_query = vectorize_query(query)
    t2 = perf_counter()
    # Add a "if" condition to use the correct collection
    logger.debug("start search by vector")
    result = search_by_vector(vector_query, collection)
    t3 = perf_counter()

    logger.info(f"Query: {query}")
    logger.info(f"Vectorization Time: {t2 - t1:.4f} seconds")
    logger.info(f"Search Time: {t3 - t2:.4f} seconds")

    return result


def get_clothes_associated_to_look(look_uuid: str, collection: weaviate.collections.classes):
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
    tenues_collection, clothes_collection = connect_collection(client)
    logger.info(tenues_collection)
    
    query = "A chic red dress"
    result = get_similar_text_to_vector(query, tenues_collection)
    logger.info(result)
    client.close()

    for o in result.objects:
        look_uuid = o.uuid
        logger.info(f"Look UUID: {look_uuid}")
        #result_clothes = get_clothes_associated_to_look(look_uuid, clothes_collection)
        #logger.info(result_clothes)