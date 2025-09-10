"""Handles Weaviate client connections and operations."""

import weaviate
import weaviate.classes.init as wvc
from weaviate.classes.query import MetadataQuery, Filter
import os
from dotenv import load_dotenv, find_dotenv
import loguru
from typing import Optional, List, Dict, Any, Union
from numpy.typing import NDArray

from tqdm import tqdm


loguru.logger.add("weaviate_{time}.log")
logger = loguru.logger


class WeaviateManager:
    """Weaviate connection manager"""

    def __init__(self):
        """Initialize Weaviate connection"""

        env_path = find_dotenv()
        load_dotenv(env_path)

        self.url = os.getenv("WEAVIATE_URL")
        self.api_key = os.getenv("WEAVIATE_KEY")

        self.headers = {
            key: value
            for key, value in {
                "X-Mistral-Api-Key": os.getenv("MISTRAL_API_KEY"),
                "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY"),
                "X-Cohere-Api-Key": os.getenv("COHERE_API_KEY"),
            }.items()
            if value is not None
        }

        self.client: Optional[weaviate.Client] = None

        if not self.url or not self.api_key:
            raise ValueError(
                "WEAVIATE_URL and WEAVIATE_KEY must be defined in .env file"
            )

        self._connect()

    def _connect(self) -> None:
        """Establish connection to Weaviate Cloud"""
        try:
            self.client = weaviate.connect_to_weaviate_cloud(
                cluster_url=self.url,
                auth_credentials=weaviate.auth.AuthApiKey(self.api_key),
                headers=self.headers,
                skip_init_checks=True,
                additional_config=wvc.AdditionalConfig(
                    timeout=wvc.Timeout(init=60, query=180, insert=180)
                ),
            )

            if self.client.is_ready():
                logger.info("✅ Weaviate connection established successfully")
            else:
                raise ConnectionError("Weaviate is not ready")

        except Exception as e:
            logger.error(f"❌ Weaviate connection error: {e}")
            raise

    def get_client(self) -> weaviate.Client:
        """Return the Weaviate client with connection validation"""
        if not self.is_connected():
            raise ConnectionError("No active connection to Weaviate")
        return self.client

    def is_connected(self) -> bool:
        """Check if connection is active and ready"""
        return self.client is not None and self.client.is_ready()

    def close(self) -> None:
        """Close the connection and cleanup resources"""
        if self.client:
            self.client.close()
            self.client = None
            logger.info("🔒 Weaviate connection closed")

    def get_collection(self, collection_name: str):
        """Get specific collection from Weaviate"""
        if not self.is_connected():
            raise ConnectionError("No active connection to Weaviate")
        return self.client.collections.use(collection_name)

    def list_collections(self) -> List[str]:
        """List all available collections"""
        if not self.is_connected():
            raise ConnectionError("No active connection to Weaviate")
        return list(self.client.collections.list_all().keys())

    def create_collection(
        self,
        collection_name: str,
        vector_config: dict = None,
        properties: List = None,
        force_creation: bool = False,
        **kwargs,
    ) -> None:
        """Create a new collection if it doesn't exist"""
        if not self.is_connected():
            raise ConnectionError("No active connection to Weaviate")

        if self.client.collections.exists(collection_name):
            if not force_creation:
                logger.warning(f"⚠️ Collection '{collection_name}' already exists")
                return
            else:
                logger.info(
                    f"🗑️ Collection '{collection_name}' exists, deleting as force_creation=True"
                )
                self.client.collections.delete(collection_name)

        create_config = {"name": collection_name}

        if vector_config:
            create_config["vector_config"] = vector_config

        if properties:
            create_config["properties"] = properties

        create_config.update(kwargs)

        self.client.collections.create(**create_config)
        logger.info(f"✅ Collection '{collection_name}' created successfully")

    def insert_object_to_collection(
        self,
        collection_name: str,
        properties: Dict[str, Any],
        vectors: Optional[Union[Dict[str, List[float]], NDArray]] = None,
        uuid: Optional[str] = None,
    ) -> str:
        """Insert a single object into a collection"""
        if not self.is_connected():
            raise ConnectionError("No active connection to Weaviate")

        collection = self.client.collections.use(collection_name)

        insert_params = {"properties": properties}
        if vectors:
            insert_params["vector"] = vectors
        if uuid:
            insert_params["uuid"] = uuid

        obj_uuid = collection.data.insert(**insert_params)
        logger.info(f"✅ Object inserted with UUID: {obj_uuid}")
        return obj_uuid

    def batch_insert_objects_to_collection(
        self,
        collection_name: str,
        objects_data: List[Dict[str, Any]],
        batch_size: Optional[int] = None,
        show_progress: bool = True,
    ) -> List[str]:
        """Batch insert multiple objects into a collection"""
        if not self.is_connected():
            raise ConnectionError("No active connection to Weaviate")

        collection = self.client.collections.get(collection_name)
        inserted_uuids = []

        # Choose batch strategy
        batch_context = (
            collection.batch.fixed_size(batch_size)
            if batch_size
            else collection.batch.dynamic()
        )

        with batch_context as batch:
            for obj_data in tqdm(objects_data, disable=not show_progress):
                # Separate properties and vectors from the combined data

                properties = obj_data.get("properties", {})
                vectors = obj_data.get("vectors")
                uuid = obj_data.get("uuid")

                # Build insert parameters (same logic as individual insert)
                batch_params = {"properties": properties}
                if vectors:
                    batch_params["vector"] = vectors
                if uuid:
                    batch_params["uuid"] = uuid

                # Add to batch
                obj_uuid = batch.add_object(**batch_params)
                inserted_uuids.append(obj_uuid)

        logger.info(f"✅ Batch inserted {len(inserted_uuids)} objects")
        return inserted_uuids

    def search_by_vector(
        self,
        query_vector: Union[List[float], NDArray],
        collection_name: str,
        target_vector: str,
        limit: int = 5,
        certainty: float = 0.0,
    ):
        """Search collection by vector similarity"""
        if not self.is_connected():
            raise ConnectionError("No active connection to Weaviate")

        collection = self.client.collections.use(collection_name)
        results = collection.query.near_vector(
            near_vector=query_vector,
            target_vector=target_vector,
            limit=limit,
            certainty=certainty,
            include_vector=True,
            return_metadata=MetadataQuery(certainty=True, distance=True),
        )

        return results

    def search_by_text(self, collection_name: str, query: str, limit: int = 5):
        """Search collection by text similarity - (only for embedded snowflake descriptions)"""
        if not self.is_connected():
            raise ConnectionError("No active connection to Weaviate")

        collection = self.client.collections.use(collection_name)
        results = collection.query.near_text(
            query=query,
            return_metadata=MetadataQuery(certainty=True, distance=True),
        )
        return results

    def search_hybrid(
        self,
        collection_name: str,
        query: str,
        query_properties: List[str],
        vector: List[float],
        target_vector: str,
        limit: int = 5,
        alpha: float = 0.5,
    ):
        """Hybrid search collection by text and vector similarity - (only for embedded snowflake descriptions)"""
        if not self.is_connected():
            raise ConnectionError("No active connection to Weaviate")

        collection = self.client.collections.use(collection_name)
        results = collection.query.hybrid(
            query=query,
            vector=vector,
            target_vector=target_vector,
            alpha=alpha,
            limit=limit,
            include_vector=True,
            return_metadata=MetadataQuery(certainty=True, distance=True),
        )
        return results

    def get_properties_of_collection(self, collection_name: str) -> List[str]:
        """Get properties of a specific collection"""
        if not self.is_connected():
            raise ConnectionError("No active connection to Weaviate")

        collection = self.client.collections.use(collection_name)
        schema = collection.config.get()
        properties = [prop.name for prop in schema.properties]

        return properties

    def query_item(
        self,
        collection_name: str,
        query_property: str,
        query_value: str,
        sub_properties: Optional[List[str]] = None,
    ):
        """Query items in a collection based on a property value"""
        if not self.is_connected():
            raise ConnectionError("No active connection to Weaviate")

        collection = self.client.collections.use(collection_name)
        result = collection.query.get(
            where={
                "path": [query_property],
                "operator": "Equal",
                "valueString": query_value,
            },
            properties=["*"] if sub_properties is None else sub_properties,
        )

        return result

    def query_item_by_fetch(
        self,
        collection_name: str,
        query_property: str,
        query_value: str,
        limit: int = 5,
    ):
        """Query items in a collection based on a property value"""
        if not self.is_connected():
            raise ConnectionError("No active connection to Weaviate")
        collection = self.client.collections.use(collection_name)

        result = collection.query.fetch_objects(
            filters=Filter.by_property(query_property).equal(query_value),
            limit=3,
            include_vector=True,
        )

        return result

    def query_item_by_id(
        self,
        collection_name: str,
        id: str,
    ):
        """Query items in a collection based on a property value"""
        if not self.is_connected():
            raise ConnectionError("No active connection to Weaviate")

        collection = self.client.collections.use(collection_name)
        result = collection.query.fetch_object_by_id(
            id,
        )

        return result

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic cleanup"""
        self.close()
