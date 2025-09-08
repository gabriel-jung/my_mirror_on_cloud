from itertools import chain
import logging
import pandas as pd
from tqdm.notebook import tqdm
from weaviate.util import generate_uuid5
import weaviate.classes.config as wc

from my_mirror_on_cloud import weaviate_manager as wm
import my_mirror_on_cloud.vector_store as vs

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_unique_keys(list_of_dicts):
    """Get all unique keys from a list of dictionaries."""
    unique_keys = set(chain.from_iterable(d.keys() for d in list_of_dicts))
    return list(unique_keys)


def process_catalog_data(catalog_data):
    """Process catalog data into structured format for Weaviate."""
    logger.info(f"🔄 Processing {len(catalog_data)} catalog items")

    df = pd.DataFrame(catalog_data)
    df["uuid"] = df["image_path"].apply(generate_uuid5)

    all_data = []
    for row in tqdm(df.itertuples(), total=len(df), desc="Processing catalog data"):
        item_data = {
            "uuid": row.uuid,
            "image_path": row.image_path,
        }

        # Process embeddings
        for embedding in row.embeddings:
            model_name = embedding["model_name"].replace("-", "")
            item_data.update(
                {
                    f"vector_{model_name}": embedding["embedding"],
                    f"timestamp_{model_name}": embedding["timestamp"],
                    f"confidence_{model_name}": embedding["confidence"],
                }
            )

        # Process tags (commented out - uncomment if needed)
        # for tag in row.tags:
        #     model_name = tag['model_name'].replace("-", "")
        #     item_data.update({
        #         f"description_{model_name}": tag['embedding'],
        #         f"timestamp_{model_name}": tag['timestamp'],
        #         f"confidence_{model_name}": tag['confidence'],
        #     })

        all_data.append(item_data)

    logger.info(f"✅ Processed {len(all_data)} items with embeddings")
    return all_data


def format_data_for_weaviate(all_data):
    """Format data for Weaviate batch insertion."""
    logger.info(f"📋 Formatting {len(all_data)} items for Weaviate")

    unique_keys = get_unique_keys(all_data)
    vector_keys = [key for key in unique_keys if key.startswith("vector_")]
    nonvector_keys = [
        key for key in unique_keys if not key.startswith("vector_") and key != "uuid"
    ]

    logger.info(f"🔍 Found {len(vector_keys)} vector keys: {vector_keys}")
    logger.info(f"📝 Found {len(nonvector_keys)} property keys: {nonvector_keys}")

    formatted_data = []
    for item in all_data:
        item_data = {"vector": {}, "properties": {}, "uuid": None}

        # Add vectors
        for key in vector_keys:
            if key in item and item[key] is not None:
                item_data["vector"][key] = item[key]

        # Add properties
        for key in nonvector_keys:
            if key in item and item[key] is not None:
                item_data["properties"][key] = item[key]

        # Add UUID
        if "uuid" in item and item["uuid"] is not None:
            item_data["uuid"] = item["uuid"]

        formatted_data.append(item_data)

    logger.info(f"✅ Formatted {len(formatted_data)} items for batch insertion")
    return formatted_data, vector_keys


def create_and_populate_collection(
    formatted_data, vector_keys, collection_name="Catalogue_HM"
):
    """Create Weaviate collection and populate with data."""
    logger.info(
        f"🏗️ Creating collection '{collection_name}' with {len(vector_keys)} vector configurations"
    )

    with wm.WeaviateManager() as weaviate:
        # Create collection
        weaviate.create_collection(
            collection_name=collection_name,
            force_creation=True,
            properties=[
                wc.Property(name="image_path", data_type=wc.DataType.TEXT),
                wc.Property(name="timestamp_fashionclip", data_type=wc.DataType.TEXT),
                wc.Property(
                    name="confidence_fashionclip", data_type=wc.DataType.NUMBER
                ),
            ],
            vector_config=[
                wc.Configure.Vectors.self_provided(
                    name=key,
                    vector_index_config=wc.Configure.VectorIndex.hnsw(
                        distance_metric=wc.VectorDistances.COSINE
                    ),
                )
                for key in vector_keys
            ],
        )

        logger.info(
            f"📦 Starting batch insertion of {len(formatted_data)} objects (batch size: 500)"
        )

        # Batch insert data
        weaviate.batch_insert_objects_to_collection(
            collection_name=collection_name,
            objects_data=formatted_data,
            batch_size=500,
            show_progress=True,
        )


def main():
    """Main function to orchestrate the catalog migration process."""
    db_path = "../data/catalogue_v1.db"
    logger.info(f"📂 Reading catalog data from: {db_path}")

    # Load catalog data
    catalog_store = vs.LocalCatalogStore(db_path=db_path)
    catalog_data = catalog_store.get_all_images()
    catalog_store.close()

    logger.info(f"📊 Loaded {len(catalog_data)} items from catalog database")

    # Process and format data
    all_data = process_catalog_data(catalog_data)
    formatted_data, vector_keys = format_data_for_weaviate(all_data)

    # Create collection and insert data
    create_and_populate_collection(formatted_data, vector_keys)

    logger.info(
        f"🎉 Successfully migrated {len(formatted_data)} items to Weaviate collection"
    )


if __name__ == "__main__":
    main()
