"""Handles local catalogue operations using SQLite."""

import sqlite3
import numpy as np
import hashlib
import io
import json

from typing import Optional, List
from tqdm import tqdm


def get_image_hash(file_path):
    """Compute SHA256 hash of an image file."""
    with open(file_path, "rb") as f:
        data = f.read()
    return hashlib.sha256(data).hexdigest()


def serialize_embeddings(embeddings_list: List) -> Optional[sqlite3.Binary]:
    """Serialize list of embedding dictionaries to binary blob."""
    if not embeddings_list:
        return None

    buffer = io.BytesIO()
    np.save(buffer, np.array(len(embeddings_list), dtype=np.int32))

    for record in embeddings_list:
        # Serialize model_name
        model_name_bytes = record["model_name"].encode("utf-8")
        np.save(buffer, np.array(len(model_name_bytes), dtype=np.int32))
        buffer.write(model_name_bytes)

        # Serialize confidence
        confidence = record.get("confidence", 1.0)
        np.save(buffer, np.array(confidence, dtype=np.float32))

        # Serialize timestamp
        timestamp_bytes = record.get("timestamp", "").encode("utf-8")
        np.save(buffer, np.array(len(timestamp_bytes), dtype=np.int32))
        buffer.write(timestamp_bytes)

        # Serialize embedding vector
        np.save(buffer, record["embedding"].astype(np.float32))

    return sqlite3.Binary(buffer.getvalue())


def deserialize_embeddings(blob_data: sqlite3.Binary) -> List:
    """Deserialize blob back to list of embedding dictionaries."""
    if blob_data is None:
        return []

    buffer = io.BytesIO(blob_data)
    embeddings = []
    n = int(np.load(buffer))

    for _ in range(n):
        # Read model_name
        name_len = int(np.load(buffer))
        model_name = buffer.read(name_len).decode("utf-8")

        # Read confidence
        confidence = float(np.load(buffer))

        # Read timestamp
        timestamp_len = int(np.load(buffer))
        timestamp = buffer.read(timestamp_len).decode("utf-8")

        # Read embedding
        embedding = np.load(buffer)

        embeddings.append(
            {
                "model_name": model_name,
                "confidence": confidence,
                "timestamp": timestamp,
                "embedding": embedding,
            }
        )

    return embeddings


class LocalCatalogStore:
    def __init__(self, db_path="../data/catalogue.db"):
        self.conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        self.cursor = self.conn.cursor()
        self._init_db()

    def _init_db(self):
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT NOT NULL,
            image_hash TEXT UNIQUE NOT NULL,
            embeddings BLOB NULL,                    -- Serialized dict of embeddings
            tags TEXT DEFAULT '[]',                  -- JSON serialized list
            processing_status TEXT DEFAULT '{}',     -- JSON serialized dict
            date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        self.conn.commit()

    def is_model_processed(self, file_path, model_name):
        record = self.get_image_by_path(file_path)
        if record and record.get("processing_status", {}).get(model_name):
            return True
        return False

    def insert_image(
        self,
        file_path,
        processing,
        tags=None,
        embeddings=None,
        force_update=False,
    ):
        if tags is None:
            tags = []
        elif isinstance(tags, dict):
            tags = [tags]
        if embeddings is None:
            embeddings = []
        elif isinstance(embeddings, dict):
            embeddings = [embeddings]
        image_hash = get_image_hash(file_path)
        tags_json = json.dumps(tags)
        processing_json = json.dumps(processing)

        self.cursor.execute(
            "SELECT id, tags, processing_status, embeddings FROM images WHERE image_hash = ?",
            (image_hash,),
        )
        row = self.cursor.fetchone()

        if row is None:
            serialized_embeddings = (
                serialize_embeddings(embeddings) if embeddings else None
            )

            self.cursor.execute(
                """
                INSERT INTO images (image_path, image_hash, embeddings, tags, processing_status)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    file_path,
                    image_hash,
                    serialized_embeddings,
                    tags_json,
                    processing_json,
                ),
            )
            self.conn.commit()
            return self.cursor.lastrowid
        else:
            model_name = list(processing.keys())[0]
            image_id, current_tags_json, current_processing_json, current_embeddings = (
                row
            )

            current_processing = json.loads(current_processing_json)

            if not force_update:
                # Check if model has already processed this image
                if model_name in current_processing and current_processing[model_name]:
                    print(
                        f"Model '{model_name}' already processed image ID {image_id}. No update done."
                    )
                    return image_id

            current_tags = json.loads(current_tags_json)
            current_embeddings = deserialize_embeddings(current_embeddings)

            # Update processing status
            current_processing.update(processing)

            # Update embeddings - replace existing embeddings for same models
            if embeddings:
                new_model_names = {emb["model_name"] for emb in embeddings}

                # Remove existing embeddings for these models
                current_embeddings = [
                    emb
                    for emb in current_embeddings
                    if emb["model_name"] not in new_model_names
                ]

                # Add new embeddings
                current_embeddings.extend(embeddings)

            # Update tags - replace existing tags for same models
            if tags:
                new_tag_models = {
                    tag.get("model_name") for tag in tags if tag.get("model_name")
                }

                # Remove existing tags for these models
                current_tags = [
                    tag
                    for tag in current_tags
                    if tag.get("model_name") not in new_tag_models
                ]

                # Add new tags
                current_tags.extend(tags)

            self.cursor.execute(
                """
                UPDATE images
                SET tags = ?, processing_status = ?, embeddings = ?
                WHERE id = ?
                """,
                (
                    json.dumps(current_tags),
                    json.dumps(current_processing),
                    serialize_embeddings(current_embeddings),
                    image_id,
                ),
            )
            self.conn.commit()
        return image_id

    def get_image_by_path(self, file_path):
        image_hash = get_image_hash(file_path)

        self.cursor.execute(
            "SELECT image_path, embeddings, tags, processing_status, date_added FROM images WHERE image_hash = ?",
            (image_hash,),
        )
        row = self.cursor.fetchone()
        if row:
            (
                image_path,
                embeddings_blob,
                tags_json,
                processing_json,
                date_added,
            ) = row
            embeddings = (
                deserialize_embeddings(embeddings_blob) if embeddings_blob else {}
            )
            tags = json.loads(tags_json)
            processing_status = json.loads(processing_json)

            return {
                "image_hash": image_hash,
                "image_path": image_path,
                "embeddings": embeddings,
                "tags": tags,
                "processing_status": processing_status,
                "date_added": date_added.isoformat(),
            }
        else:
            return None

    def get_all_images(self, show_progress: bool = True) -> List[dict]:
        self.cursor.execute(
            "SELECT image_hash, image_path, embeddings, tags, processing_status, date_added FROM images"
        )
        rows = self.cursor.fetchall()
        results = []
        for row in tqdm(rows, disable=not show_progress):
            (
                image_hash,
                image_path,
                embeddings_blob,
                tags_json,
                processing_json,
                date_added,
            ) = row

            embeddings = (
                deserialize_embeddings(embeddings_blob) if embeddings_blob else {}
            )
            tags = json.loads(tags_json)
            processing_status = json.loads(processing_json)
            results.append(
                {
                    "image_hash": image_hash,
                    "image_path": image_path,
                    "embeddings": embeddings,
                    "tags": tags,
                    "processing_status": processing_status,
                    "date_added": date_added.isoformat(),
                }
            )
        return results

    def get_all_columns(self):
        """Get all table names and their column names."""
        # Get table names
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = self.cursor.fetchall()

        result = {}

        # For each table, get column names
        for table in tables:
            table_name = table[0]
            self.cursor.execute(f"PRAGMA table_info({table_name});")
            columns = self.cursor.fetchall()
            column_names = [col[1] for col in columns]
            result[table_name] = column_names

        return result

    def close(self):
        self.conn.close()
