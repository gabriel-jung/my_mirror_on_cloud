"""Catalogue search module on Weaviate's vector database."""

import loguru
import numpy as np
from collections import defaultdict
from itertools import product
from time import perf_counter

from .weaviate_manager import WeaviateManager
from .embedding_manager import vectorize_texts

logger = loguru.logger

# def weaviate_connect() -> weaviate.Client:
#     """
#     Connect to Weaviate instance.
#     output: Weaviate client
#     """
#     load_dotenv(override=True)

#     # Open a Weaviate client
#     client = weaviate.connect_to_weaviate_cloud(
#         cluster_url=WEAVIATE_URL,
#         auth_credentials=Auth.api_key(WEAVIATE_KEY),

#         headers = {
#             "X-Mistral-Api-Key": MISTRAL_API_KEY
#         },
#     )

#     if client.is_ready():
#         logger.info("Weaviate client is ready")
#         return client
#     else:
#         logger.error("Weaviate client is not ready")
#         return None


# def connect_collection(client: weaviate.Client):
#     """
#     Connect to the Weaviate collection and load the CLIP model and tokenizer.
#     input: Weaviate client
#     output: Weaviate collection, CLIP model, CLIP tokenizer, CLIP processor
#     """
#     tenues_collection = client.collections.use("ImageCatalogue")
#     clothes_collection = client.collections.use("Vetements")

#     return tenues_collection, clothes_collection


# def search_by_vector(vector_query: np.ndarray, collection):
#     """
#     Search for look/items corresponding to the text query.
#     input: vector of an image or text, Weaviate collection
#     output: Weaviate query result
#     """
#     logger.debug(collection)
#     logger.debug(vector_query)
#     result = collection.query.near_vector(
#         near_vector = vector_query,
#         limit=3,
#         include_vector=True,
#         return_metadata=MetadataQuery(certainty=True)
#     )
#     logger.debug(result)

#     for o in result.objects:
#         logger.info(o.properties)
#         # similarity
#         logger.info(o.metadata.certainty)
#         #logger.info(o.vector["default"])
#         #logger.info(o.uuid)

#     return result


# def get_similar_text_to_vector(query: str, collection):
#     """
#     Get the most similar look/item to the text query.
#     input: text query, Weaviate collection
#     output: Weaviate query result
#     """

#     t1 = perf_counter()
#     logger.debug("start vectorize_query")
#     vector_query = vectorize_query(query)
#     t2 = perf_counter()
#     # Add a "if" condition to use the correct collection
#     logger.debug("start search by vector")
#     result = search_by_vector(vector_query, collection)
#     t3 = perf_counter()

#     logger.info(f"Query: {query}")
#     logger.info(f"Vectorization Time: {t2 - t1:.4f} seconds")
#     logger.info(f"Search Time: {t3 - t2:.4f} seconds")

#     return result


# def get_clothes_associated_to_look(
#     client: weaviate.Client, look_uuid: str, look_collection: str = "Look"
# ):
#     """
#     Get clothes associated to a look.

#     :param client: Weaviate client instance
#     :param look_uuid: UUID of the look
#     :param look_collection: Collection name where looks are stored
#     :return: Query result with clothes data
#     """
#     collection = client.collections.use(look_collection)

#     result = collection.query.get(
#         where={
#             "path": ["tenue_uuid"],  # or "uuid" depending on your schema
#             "operator": "Equal",
#             "valueString": look_uuid,
#         },
#         properties=[
#             "name",
#             "description",
#             "image_url",
#             "clothes { name description image_url price link }",
#         ],
#     )

#     for obj in result.objects:
#         logger.info(f"Look UUID: {obj.uuid}")
#         logger.info(f"Look properties: {obj.properties}")

#     return result


# def get_clothes_associated_to_look(look_uuid: str, look_collection: str = "Look"):
#     """
#     Get clothes associated to a look using WeaviateManager.

#     :param weaviate_manager: WeaviateManager instance
#     :param look_uuid: UUID of the look
#     :param look_collection: Collection name where looks are stored
#     :return: Query result with clothes data
#     """
#     with WeaviateManager() as wm:
#         result = wm.query_item(
#             collection_name=look_collection,
#             query_property="tenue_uuid",  # or "uuid" depending on your schema
#             query_value=look_uuid,
#             sub_properties=[
#                 "name",
#                 "description",
#                 "image_url",
#                 "clothes { name description image_url price link }",
#             ],
#         )

#         headers = {
#             "X-Mistral-Api-Key": MISTRAL_API_KEY
#         },
#     )

#     if client.is_ready():
#         logger.info("Weaviate client is ready")
#         return client
#     else:
#         logger.error("Weaviate client is not ready")
#         return None



def connect_collection():
    """
    Connect to the Weaviate collection and load the CLIP model and tokenizer.
    input: Weaviate client
    output: Weaviate collection, CLIP model, CLIP tokenizer, CLIP processor 
    """
    # with WeaviateManager() as wm:
        # tenues_collection = wm.client.collections.use("Tenues_v2025_dual")
        # clothes_collection = wm.client.collections.use("Vetements_v2025_dual")
        # catalogue_collection = wm.client.collections.use("Test_collection_farfetch2")
    tenues_collection = "Tenues_v2025_dual"
    clothes_collection = "Vetements_v2025_dual"
    catalogue_collection = "Test_collection_farfetch2"
    return tenues_collection, clothes_collection, catalogue_collection



# def search_by_vector(vector_query: np.ndarray, collection, target_vector="default"):
#     """
#     Search for look/items corresponding to the text query.
#     input: vector of an image or text, Weaviate collection
#     output: Weaviate query result
#     """
#     return collection.query.near_vector(
#         near_vector = vector_query,
#         limit=3,
#         include_vector=True,
#         target_vector=target_vector,
#         return_metadata=MetadataQuery(certainty=True)
#     )


def get_similar_text_to_vector(query: str, collection):
    """
    Get the most similar look/item to the text query.
    input: text query, Weaviate collection
    output: Weaviate query result
    """
    # logger.debug("start vectorize_query")
    # vector_query = vectorize_query(query)
    # # Add a "if" condition to use the correct collection
    # logger.debug("start search by vector")
    # result = search_by_vector(vector_query, collection, "fclip")
    
    t1 = perf_counter()
    query_vector = vectorize_texts([query], model_name="fashion-clip")[0]["embedding"]
    t2 = perf_counter()
    with WeaviateManager() as wm:
        result = wm.search_by_vector(
            query_vector=query_vector,
            collection_name=collection,
            target_vector="fclip",
            limit=3,
            #certainty=0.7,
        )
        t3 = perf_counter()
        logger.info(f"Query: {query}")
        logger.info(f"Vectorization Time (text): {t2 - t1:.4f} seconds")
        logger.info(f"Search Time: {t3 - t2:.4f} seconds")

        return result


def get_clothes_associated_to_look(tenueId: str, certainty: float, collection):
    """
    Get the clothes associated to a look.
    input: look uuid, Weaviate collection
    output: Weaviate query result
    """  
    t1 = perf_counter()
    # result = collection.query.fetch_objects(
    #     filters=Filter.by_property("origImageId").equal(tenueId),
    #     limit=3,
    #     include_vector=True
    # )          
    with WeaviateManager() as wm:
        result = wm.query_item_by_fetch(
            collection_name=collection,
            query_property="origImageId",
            query_value=tenueId,
            limit=3,
        )               
    
    reco_list = []
    # results contient la liste des items correspondants
    for item in result.objects:
        reco_list.append({
            "tenueId": tenueId,
            "categoryName": item.properties['categoryName'],                      
            "imageId": item.properties['imageCroppedId'],
            "vectorFClip": item.vector['fclip'],
            "certainty": certainty
        })

    t2 = perf_counter()
    logger.info(f"fetch clothes: {t2 - t1:.4f} seconds")
    return reco_list


def get_similar_vector_to_vector(reco_list: np.array, collection):
    # get similar item from outfit's item reference
    #tenue_id = None
    similar_clothes_list = []
    t1 = perf_counter()
    for outfit in reco_list:
        for item in outfit:
            with WeaviateManager() as wm:
                similar_clothes = wm.search_by_vector(item["vectorFClip"], collection, "vector_fashionclip")
                for cloth in similar_clothes.objects:
                    similar_clothes_list.append({
                        "tenueId": item["tenueId"],
                        "imageId": item["imageId"],
                        "cloth_path": cloth.properties['image_name'],
                        "certainty": (cloth.metadata.certainty + item["certainty"]) /2
                    })
    t2 = perf_counter()
    logger.info(f"get similar clothes: {t2 - t1:.4f} seconds")
    # Calculate best combination among outfit
    unique_tenueIds = list({item["tenueId"] for item in similar_clothes_list})
    results = []

    for tenuesId in unique_tenueIds:
        filtered_data = [item for item in similar_clothes_list if item["tenueId"] == tenuesId]
        indices_by_id = defaultdict(list)
        for i, item in enumerate(filtered_data):
            indices_by_id[item["imageId"]].append(i)
        #logger.info(f"indices_by_id: {indices_by_id}")

        # créer toutes les combinaisons : un élément par id
        all_combinations = product(*indices_by_id.values())

        # 3. Calculer totalcertainty 
        for combo in all_combinations:
            certainties = [filtered_data[i]["certainty"] for i in combo]
            paths = [filtered_data[i]["cloth_path"] for i in combo]
            total_certainty = sum(certainties)/len(certainties)

            results.append({"combo": combo, "cloth_path": paths, "totalcertainty": total_certainty})

    # Get best recommended outfit
    best_outfit = sorted(results, key=lambda x: x["totalcertainty"], reverse=True)[:3]

    t3 = perf_counter()
    logger.info(f"get best outfit: {t3 - t2:.4f} seconds")
    return best_outfit



def algo_flow(query, tenues_col, vet_col, cat_col )-> np.array:

    cleaned_query = query.lower().split(":",1)[1].strip()
    logger.info(f"Cleaned query: {cleaned_query}")

    if "flow1" in query:
        # search corresponding outfit in trendy outfit database
        matching_outfit = get_similar_text_to_vector(cleaned_query, tenues_col)
        logger.info(f"Matching outfit found: {matching_outfit}")
        reco_list = []
        # Get the items from the selected outfit
        for o in matching_outfit.objects:
            tenue_id = o.properties['tenueId']
            reco_list.append(get_clothes_associated_to_look(tenue_id, o.metadata.certainty, vet_col))

        # Search in catalogue the similar clothing items (by vectorFClip)
        best_outfit = get_similar_vector_to_vector(reco_list, cat_col)

        recommended_items = best_outfit


    #elif "flow2" in query:
        # Get the vector description of the existing clothing item
        #vectorized_image = vectorize_image(image) if image else None

        # search outfit in trendy outfit database (hybrid search)

        # Get 3 matching outfits

        # Get the vector description of the wanted clothing item

        # Search in catalogue

    elif "flow3" in query:
        # Search directly in catalogue
        result = get_similar_text_to_vector(cleaned_query, cat_col)
        recommended_items = result

    else:
        recommended_items = None
        
    return recommended_items


if __name__ == "__main__":
    #     client = weaviate_connect()
    #     tenues_collection, clothes_collection = connect_collection(client)
    #     logger.info(tenues_collection)
    ## Gabriel
    

    
    ## My code
    t1 = perf_counter()
    tenues_collection, clothes_collection, catalogue_collection = connect_collection()
    #get_similar_vector_to_vector([], clothes_collection)
    t2 = perf_counter()
    logger.info(f"connect to weaviate: {t2 - t1:.4f} seconds")


    query = "flow1: a casual woman outfit for summer"
    result = algo_flow(query, tenues_collection, clothes_collection, catalogue_collection)
    logger.info(result)
    #     result = get_similar_text_to_vector(query, tenues_collection)
    #     logger.info(result)
    #     client.close()