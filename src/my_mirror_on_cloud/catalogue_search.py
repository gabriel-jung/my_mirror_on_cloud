"""Catalogue search module on Weaviate's vector database."""

import loguru
import numpy as np
from collections import defaultdict
from itertools import product
from time import perf_counter

from .weaviate_manager import WeaviateManager
from .embedding_manager import vectorize_texts

logger = loguru.logger

def connect_collection():
    """
    Connect to the Weaviate collection and load the CLIP model and tokenizer.
    input: Weaviate client
    output: Weaviate collection, CLIP model, CLIP tokenizer, CLIP processor 
    """
    tenues_collection = "Tenues_v2025_dual"
    clothes_collection = "Vetements_v2025_dual"
    catalogue_collection = "Catalogue_HM"
    return tenues_collection, clothes_collection, catalogue_collection


def get_similar_text_to_vector(query: str, collection, wm, fashion_clip_emb, tags):
    """
    Get the most similar look/item to the text query.
    input: text query, Weaviate collection
    output: Weaviate query result
    """    
    t1 = perf_counter()
    query_vector = vectorize_texts(fashion_clip_emb, [query], model_name="fashion-clip")[0]["embedding"]
    t2 = perf_counter()
    # with WeaviateManager() as wm:
    target_vector = "embedding_fashionclip" if collection == "Catalogue_HM" else "fclip"
    result = wm.search_by_vector(
        query_vector=query_vector,
        collection_name=collection,
        target_vector=target_vector,
        limit=3,
        #certainty=0.7,
    )
    # hybride search
    
    t3 = perf_counter()
    logger.info(f"Query: {query}")
    logger.info(f"Vectorization Time (text): {t2 - t1:.4f} seconds")
    logger.info(f"Search Time: {t3 - t2:.4f} seconds")

    return result


def get_clothes_associated_to_look(tenueId: str, certainty: float, collection, wm):
    """
    Get the clothes associated to a look.
    input: look uuid, Weaviate collection
    output: Weaviate query result
    """  
    t1 = perf_counter()
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


def get_similar_vector_to_vector(reco_list: np.array, collection, wm):
    # get similar item from outfit's item reference
    similar_clothes_list = []
    t1 = perf_counter()
    for outfit in reco_list:
        for item in outfit:
            similar_clothes = wm.search_by_vector(item["vectorFClip"], collection, "embedding_fashionclip")
            for cloth in similar_clothes.objects:
                similar_clothes_list.append({
                    "tenueId": item["tenueId"],
                    "imageId": item["imageId"],
                    "cloth_path": cloth.properties['image_path'],
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



def algo_flow(wm, query, tenues_col, vet_col, cat_col, fashion_clip_emb )-> np.array:
    t1 = perf_counter()
    logger.info(query)
    tags = query[1:]
    logger.info(tags)
    if query != ["Need clarification"]:
        cleaned_query = query[0].lower().split(":",1)[1].strip()
        logger.info(f"Cleaned query: {cleaned_query}")
    
    if "flow1" in query[0]:
        # search corresponding outfit in trendy outfit database
        matching_outfit = get_similar_text_to_vector(cleaned_query, tenues_col, wm, fashion_clip_emb)
        logger.info(f"Matching outfit found: {matching_outfit}")
        reco_list = []
        # Get the items from the selected outfit
        for o in matching_outfit.objects:
            tenue_id = o.properties['tenueId']
            reco_list.append(get_clothes_associated_to_look(tenue_id, o.metadata.certainty, vet_col, wm))

        # Search in catalogue the similar clothing items (by vectorFClip)
        best_outfit = get_similar_vector_to_vector(reco_list, cat_col, wm)

        recommended_items = best_outfit


    #elif "flow2" in query[0]:
        # Get the vector description of the existing clothing item
        #vectorized_image = vectorize_image(image) if image else None

        # search outfit in trendy outfit database (hybrid search)

        # Get 3 matching outfits

        # Get the vector description of the wanted clothing item

        # Search in catalogue

    elif "flow3" in query[0]:
        # Search directly in catalogue
        result = get_similar_text_to_vector(cleaned_query, cat_col, wm, fashion_clip_emb, tags)
        reco_list=[]
        for item in result.objects:
            reco_list.append({                    
                "cloth_path": [item.properties['image_path']],
                "totalcertainty": item.metadata.certainty
            })

        
        recommended_items = reco_list

    else:
        recommended_items = None
    t2 =perf_counter()
    logger.info(f"######## TOTAL Time: {t2 - t1:.4f} seconds #######")
    return recommended_items


if __name__ == "__main__":
    t1 = perf_counter()
    tenues_collection, clothes_collection, catalogue_collection = connect_collection()
    t2 = perf_counter()
    logger.info(f"connect to weaviate: {t2 - t1:.4f} seconds")

    query = "flow1: a casual woman outfit for summer"
    result = algo_flow(query, tenues_collection, clothes_collection, catalogue_collection)
    logger.info(result)
