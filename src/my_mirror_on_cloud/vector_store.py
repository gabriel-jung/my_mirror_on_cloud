"""Handles local catalog operations using SQLite and Weaviate."""

import sqlite3
import numpy as np
import hashlib
import io
import json


def get_image_hash(file_path):
    """Compute SHA256 hash of an image file."""
    with open(file_path, "rb") as f:
        data = f.read()
    return hashlib.sha256(data).hexdigest()


def adapt_array(arr):
    """Convert numpy array to binary for SQLite storage."""
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def convert_array(blob):
    """Convert binary from SQLite back to numpy array."""
    out = io.BytesIO(blob)
    out.seek(0)
    return np.load(out)


sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("array", convert_array)


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
            embedding array NULL,
            tags TEXT DEFAULT '[]',                -- JSON serialized list
            processing_status TEXT DEFAULT '{}',   -- JSON serialized dict
            date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        self.conn.commit()

    def is_model_processed(self, file_path, model_name):
        record = self.get_image_by_path(file_path)
        if record and record.get("processing_status", {}).get(model_name):
            return True
        return False

    def insert_image(self, file_path, tags_dict, processing_dict, embedding=None):
        if isinstance(tags_dict, dict):
            tags_dict = [tags_dict]
        image_hash = get_image_hash(file_path)
        tags_json = json.dumps(tags_dict)
        processing_json = json.dumps(processing_dict)

        self.cursor.execute(
            "SELECT id, tags, processing_status, embedding FROM images WHERE image_hash = ?",
            (image_hash,),
        )
        row = self.cursor.fetchone()

        if row is None:
            self.cursor.execute(
                """
                INSERT INTO images (image_path, image_hash, embedding, tags, processing_status)
                VALUES (?, ?, ?, ?, ?)
                """,
                (file_path, image_hash, embedding, tags_json, processing_json),
            )
            self.conn.commit()
            return self.cursor.lastrowid
        else:
            model_name = tags_dict[0]["model_name"]
            image_id, current_tags_json, current_processing_json, current_embedding = (
                row
            )

            current_tags = json.loads(current_tags_json)
            current_processing = json.loads(current_processing_json)

            # Check if model has already processed this image
            if model_name in current_processing and current_processing[model_name]:
                print(
                    f"Model '{model_name}' already processed image ID {image_id}. No update done."
                )
                return image_id

            # Update processing status
            current_processing.update(processing_dict)

            # Merge tags: simple example appends new tags (customize as needed)
            updated_tags = current_tags + tags_dict

            # Decide whether to update embedding (e.g., if new embedding provided)
            updated_embedding = (
                embedding if embedding is not None else current_embedding
            )

            self.cursor.execute(
                """
                UPDATE images
                SET tags = ?, processing_status = ?, embedding = ?
                WHERE id = ?
                """,
                (
                    json.dumps(updated_tags),
                    json.dumps(current_processing),
                    updated_embedding,
                    image_id,
                ),
            )
            self.conn.commit()
        return image_id

    def update_embedding(self, file_path, embedding):
        image_hash = get_image_hash(file_path)

        self.cursor.execute(
            "UPDATE images SET embedding = ? WHERE image_hash = ?",
            (embedding, image_hash),
        )
        if self.cursor.rowcount == 0:
            print("No image found with given hash to update embedding.")
        else:
            self.conn.commit()

    def get_image_by_path(self, file_path):
        image_hash = get_image_hash(file_path)

        self.cursor.execute(
            "SELECT id, image_path, embedding, tags, processing_status, date_added FROM images WHERE image_hash = ?",
            (image_hash,),
        )
        row = self.cursor.fetchone()
        if row:
            image_id = row[0]
            image_path = row[1]
            embedding = row[2]  # numpy array, auto converted
            tags = json.loads(row[3])
            processing_status = json.loads(row[4])
            date_added = row[5]
            return {
                "id": image_id,
                "image_path": image_path,
                "embedding": embedding,
                "tags": tags,
                "processing_status": processing_status,
                "date_added": date_added.isoformat(),
            }
        else:
            return None

    def get_all_images(self):
        self.cursor.execute(
            "SELECT id, image_path, embedding, tags, processing_status, date_added FROM images"
        )
        rows = self.cursor.fetchall()
        results = []
        for row in rows:
            results.append(
                {
                    "id": row[0],
                    "image_path": row[1],
                    "embedding": row[2],
                    "tags": json.loads(row[3]),
                    "processing_status": json.loads(row[4]),
                    "date_added": row[5].isoformat(),
                }
            )
        return results

    def close(self):
        self.conn.close()
