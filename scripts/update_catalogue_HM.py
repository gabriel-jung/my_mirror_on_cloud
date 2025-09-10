from itertools import chain
import pandas as pd
from tqdm.notebook import tqdm
from weaviate.util import generate_uuid5
import weaviate.classes.config as wc

from my_mirror_on_cloud import weaviate_manager as wm
import my_mirror_on_cloud.vector_store as vs
from my_mirror_on_cloud.utils import clean_name

from loguru import logger


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

        for embedding in row.embeddings:
            model_suffix = (
                f"_{embedding['model_name']}" if "model_name" in embedding else ""
            )
            for key, value in embedding.items():
                if key != "model_name":
                    item_data[clean_name(f"{key}{model_suffix}")] = value
        for tag in row.tags:
            model_suffix = f"_{tag['model_name']}" if "model_name" in tag else ""
            for key, value in tag.items():
                if key != "model_name":
                    item_data[clean_name(f"{key}{model_suffix}")] = value

        all_data.append(item_data)

    logger.info(f"✅ Processed {len(all_data)} items with embeddings")
    return all_data


def format_data_for_weaviate(all_data):
    """Format data for Weaviate batch insertion."""
    logger.info(f"📋 Formatting {len(all_data)} items for Weaviate")

    unique_keys = get_unique_keys(all_data)
    vector_keys = [
        clean_name(key) for key in unique_keys if key.startswith("embedding_")
    ]
    nonvector_keys = [
        clean_name(key)
        for key in unique_keys
        if not key.startswith("embedding_") and key != "uuid"
    ]

    logger.info(f"🔍 Found {len(vector_keys)} vector keys: {vector_keys}")
    logger.info(f"📝 Found {len(nonvector_keys)} property keys: {nonvector_keys}")

    formatted_data = []
    for item in all_data:
        item_data = {"vectors": {}, "properties": {}, "uuid": None}

        # Add vectors
        for key in vector_keys:
            if key in item and item[key] is not None:
                item_data["vectors"][key] = item[key]

        # Add properties
        for key in nonvector_keys:
            if key in item and item[key] is not None:
                item_data["properties"][key] = item[key]

        # Add UUID
        if "uuid" in item and item["uuid"] is not None:
            item_data["uuid"] = item["uuid"]

        formatted_data.append(item_data)

    logger.info(f"✅ Formatted {len(formatted_data)} items for batch insertion")
    return formatted_data, vector_keys, nonvector_keys


def create_dynamic_properties(nonvector_keys):
    """Create properties dynamically based on available keys."""
    properties = []

    for key in nonvector_keys:
        if key.startswith("timestamp_"):
            properties.append(wc.Property(name=key, data_type=wc.DataType.DATE))
        elif key.startswith("confidence_"):
            properties.append(wc.Property(name=key, data_type=wc.DataType.NUMBER))
        else:
            properties.append(wc.Property(name=key, data_type=wc.DataType.TEXT))

    return properties


def create_dynamic_vector_configs(vector_keys, nonvector_keys=[]):
    """Create vector configurations dynamically based on available keys."""
    vector_configs = []
    for key in vector_keys:
        vector_configs.append(
            wc.Configure.Vectors.self_provided(
                name=key,
                vector_index_config=wc.Configure.VectorIndex.hnsw(
                    distance_metric=wc.VectorDistances.COSINE
                ),
            )
        )
    for key in nonvector_keys:
        if "description" in key:
            model_name = key.replace("description_", "")
            vector_configs.append(
                wc.Configure.Vectors.text2vec_transformers(
                    name="embedding_description_" + model_name,
                    model="Snowflake/snowflake-arctic-embed-l-v2.0",
                    source_properties=[key],
                )
            )
    print(vector_configs)
    return vector_configs


def create_and_populate_collection(
    formatted_data, vector_keys, nonvector_keys, collection_name="Catalogue_HM"
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
            properties=create_dynamic_properties(nonvector_keys),
            vector_config=create_dynamic_vector_configs(vector_keys, nonvector_keys),
        )

        logger.info("📦 Starting batch insertion")

        # Try normal batch insertion first
        try:
            weaviate.batch_insert_objects_to_collection(
                collection_name=collection_name,
                objects_data=formatted_data,
                batch_size=100,
                show_progress=True,
            )
            logger.info("✅ Batch insertion completed successfully")

        except Exception as e:
            logger.warning(f"Some batches failed, checking for failed objects: {e}")

            # Check if there are failed objects to retry
            if hasattr(weaviate.client, "batch") and hasattr(
                weaviate.client.batch, "failed_objects"
            ):
                failed_objects = weaviate.client.batch.failed_objects
                if failed_objects:
                    logger.info(
                        f"🔄 Retrying {len(failed_objects)} failed objects with smaller batches"
                    )

                    # Retry failed objects with batch size of 10
                    weaviate.batch_insert_objects_to_collection(
                        collection_name=collection_name,
                        objects_data=failed_objects,
                        batch_size=10,
                        show_progress=True,
                    )


def main():
    """Main function to orchestrate the catalog migration process."""
    try:
        db_path = "../data/catalogue_v1.db"
        logger.info(f"📂 Reading catalog data from: {db_path}")

        # Load catalog data
        catalog_store = vs.LocalCatalogStore(db_path=db_path)
        catalog_data = catalog_store.get_all_images()
        catalog_store.close()

        if not catalog_data:
            logger.warning("No data found in catalog database")
            return

        logger.info(f"📊 Loaded {len(catalog_data)} items from catalog database")

        # Process and format data
        all_data = process_catalog_data(catalog_data)
        if not all_data:
            logger.error("No valid data after processing")
            return

        formatted_data, vector_keys, nonvectors_keys = format_data_for_weaviate(
            all_data
        )

        # Create collection and insert data
        create_and_populate_collection(formatted_data, vector_keys, nonvectors_keys)

        logger.info(
            f"🎉 Successfully migrated {len(formatted_data)} items to Weaviate collection"
        )

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise


if __name__ == "__main__":
    main()
