"""Handles embedding operations for text and images."""

import time
import numpy as np
from typing import List, Optional, Any, Union
from datetime import datetime, timezone
from pathlib import Path
from joblib import Parallel, delayed
from PIL import Image

import loguru
import timm

import torch
import torchvision.transforms as T

from fashion_clip.fashion_clip import FashionCLIP

from .utils import resize_image, apply_l2norm

logger = loguru.logger


model_confidence = {
    "fashion-clip": 0.90,
    "vit_base_patch16_224": 0.80,
}


class ClothingImageEmbedder:
    """Base class for image embedding operations."""

    def __init__(self, model_name: str = "fashion-clip"):
        self.model_name = model_name
        self.model = None
        self.l2norm = False
        self._load_model()

    def _load_model(self):
        """Load the embedding model."""
        raise NotImplementedError("Must implement _load_model")

    def encode_images(self, image_paths: Union[str, List[str]], **kwargs) -> Any:
        """Encode images to embeddings."""
        raise NotImplementedError("Must implement encode_images")

    def encode_texts(self, texts: Union[str, List[str]], **kwargs) -> Any:
        """Encode texts to embeddings."""
        raise NotImplementedError("Must implement encode_texts")

    def is_valid_embedding(self, embedding: Any) -> bool:
        """Check if embedding is valid."""
        raise NotImplementedError("Must implement is_valid_embedding")


class FashionCLIPEmbedder(ClothingImageEmbedder):
    """Handles FashionCLIP embedding operations."""

    def __init__(
        self,
        model_name: str = "fashion-clip",
        use_float16: bool = True,
        l2norm: bool = False,
    ):
        self.use_float16 = use_float16
        self.l2norm = l2norm
        super().__init__(model_name)

    def _load_model(self):
        """Load FashionCLIP model."""
        try:
            logger.info("Loading FashionCLIP model...")
            self.model = FashionCLIP("fashion-clip")

            if self.use_float16:
                self.model.model.half()
                logger.success("FashionCLIP loaded with float16")
            else:
                logger.info("FashionCLIP loaded with float32")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Failed to load FashionCLIP model: {e}")

    def encode_texts(
        self, texts: Union[str, List[str]], batch_size: int = 32
    ) -> np.ndarray:
        """Encode text to embeddings using FashionCLIP."""
        if isinstance(texts, str):
            texts = [texts]
        try:
            return self.model.encode_text(texts, batch_size=batch_size)
        except Exception as e:
            raise RuntimeError(f"Failed to encode texts: {e}")

    def encode_images(
        self,
        image_paths: Union[str, List[str]],
        batch_size: int = 32,
        max_width: Optional[int] = None,
        n_jobs: int = -1,
        **kwargs,
    ) -> np.ndarray:
        """Encode images to embeddings using FashionCLIP."""

        # Handle single image path
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        # Validate image paths
        valid_paths = []
        for path in image_paths:
            if Path(path).exists():
                valid_paths.append(path)
            else:
                raise FileNotFoundError(f"Image not found: {path}")

        # Encode images
        try:
            if max_width is None:
                return self.model.encode_images(valid_paths, batch_size=batch_size)
            embeddings = []
            for i in range(0, len(valid_paths), batch_size):
                batch_paths = valid_paths[i : i + batch_size]

                # Process all images in parallel
                preprocessed_batch = Parallel(n_jobs=n_jobs)(
                    delayed(resize_image)(path, max_width) for path in batch_paths
                )

                # Encode this preprocessed batch using official function
                batch_embeddings = self.model.encode_images(
                    preprocessed_batch, batch_size=len(preprocessed_batch)
                )
                embeddings.append(batch_embeddings)
            if self.l2norm:
                return apply_l2norm(np.vstack(embeddings))
            return np.vstack(embeddings)
        except Exception as e:
            raise RuntimeError(f"Failed to encode images: {e}")

    def is_valid_embedding(self, embedding: np.ndarray) -> bool:
        """Check if FashionCLIP embedding is valid."""
        if embedding is None:
            return False

        if not isinstance(embedding, np.ndarray):
            return False

        # Check for expected embedding dimensions (FashionCLIP: 512-dim vectors)
        if embedding.ndim not in [1, 2]:
            return False

        # Check for NaN or infinite values
        if np.isnan(embedding).any() or np.isinf(embedding).any():
            return False

        return True


class ViTEmbedder(ClothingImageEmbedder):
    """Handles Vision Transformer (ViT) embedding operations - Mac optimized."""

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        use_float16: bool = True,
        l2norm: bool = False,
    ):
        self.use_float16 = use_float16
        self.transforms = None
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.l2norm = l2norm
        super().__init__(model_name)

    def _load_model(self):
        """Load ViT model."""
        try:
            self.model = timm.create_model(
                self.model_name, pretrained=True, num_classes=0
            ).eval()

            self.model = self.model.to(self.device)

            if self.use_float16:
                try:
                    self.model = self.model.to(dtype=torch.float16)
                    logger.info(f"ViT {self.model_name} loaded with float16")
                except Exception as e:
                    self.model = self.model.to(dtype=torch.float32)
                    logger.warning(
                        f"ViT {self.model_name} loaded with float32, float16 failed: {e}"
                    )
            else:
                self.model = self.model.to(dtype=torch.float32)
                logger.info(f"ViT {self.model_name} loaded with float32")

            self.transforms = T.Compose(
                [
                    T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

            torch.set_grad_enabled(False)
        except Exception as e:
            raise RuntimeError(f"Failed to load ViT model {self.model_name}: {e}")

    @torch.inference_mode()
    def encode_images(
        self,
        image_paths: Union[str, List[str]],
        batch_size: int = 16,
        max_width: Optional[int] = None,
        **kwargs,
    ) -> np.ndarray:
        """Encode images to embeddings using ViT."""

        if isinstance(image_paths, str):
            image_paths = [image_paths]

        valid_paths = []
        for path in image_paths:
            if Path(path).exists():
                valid_paths.append(path)
            else:
                raise FileNotFoundError(f"Image not found: {path}")

        try:
            embeddings = []
            for i in range(0, len(valid_paths), batch_size):
                batch_paths = valid_paths[i : i + batch_size]
                batch_tensors = []

                for path in batch_paths:
                    if max_width is not None:
                        img = resize_image(path, max_width=max_width)
                    else:
                        img = Image.open(path).convert("RGB")

                    tensor = self.transforms(img)
                    batch_tensors.append(tensor)

                batch_tensor = torch.stack(batch_tensors).to(
                    self.device, non_blocking=True
                )

                if self.use_float16 and batch_tensor.dtype == torch.float32:
                    batch_tensor = batch_tensor.to(torch.float16)
                elif not self.use_float16 and batch_tensor.dtype != torch.float32:
                    batch_tensor = batch_tensor.to(torch.float32)

                batch_embeddings = self.model(batch_tensor).detach().cpu().numpy()
                embeddings.append(batch_embeddings)

            all_embeddings = np.vstack(embeddings)
            if self.l2norm:
                all_embeddings = apply_l2norm(all_embeddings)
            return all_embeddings

        except Exception as e:
            raise RuntimeError(f"Failed to encode images with ViT: {e}")

    def encode_texts(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """ViT doesn't support text encoding."""
        raise NotImplementedError("ViT model doesn't support text encoding")

    def is_valid_embedding(self, embedding: np.ndarray) -> bool:
        """Check if ViT embedding is valid."""
        if embedding is None:
            return False

        if not isinstance(embedding, np.ndarray):
            return False

        if embedding.ndim not in [1, 2]:
            return False

        expected_dim = 768
        if embedding.ndim == 1 and embedding.shape[0] != expected_dim:
            return False
        elif embedding.ndim == 2 and embedding.shape[1] != expected_dim:
            return False

        if np.isnan(embedding).any() or np.isinf(embedding).any():
            return False

        return True


def create_embedder(
    model_name: str = "fashion-clip", **kwargs
) -> ClothingImageEmbedder:
    """Factory function to create appropriate embedder."""

    if model_name == "fashion-clip":
        return FashionCLIPEmbedder(model_name, **kwargs)
    elif model_name in ["vit_base_patch16_224", "vit_large_patch16_224"]:
        return ViTEmbedder(model_name, **kwargs)
    else:
        raise ValueError(f"Unsupported embedding model: {model_name}")


def vectorize_images(
    embedder: ClothingImageEmbedder,
    image_paths: Union[str, List[str]],
    model_name: str = "fashion-clip",
    batch_size: int = 32,
    max_width: Optional[int] = None,
    max_retries: int = 10,
    use_float16: bool = True,
) -> List[dict]:
    """Complete image vectorization pipeline with batch processing and individual results."""

    start_time = time.time()
    results = []

    try:
        # Create embedder
        #embedder = create_embedder(model_name, use_float16=use_float16)

        # Handle single image path
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        num_images = len(image_paths)
        embeddings = None
        is_valid = False
        error_msg = ""

        # Process all images in batch with retries
        for attempt in range(max_retries):
            try:
                embeddings = embedder.encode_images(
                    image_paths, batch_size=batch_size, max_width=max_width
                )

                is_valid = embedder.is_valid_embedding(embeddings)

                if is_valid:
                    break
                else:
                    error_msg = f"Invalid embeddings generated on attempt {attempt + 1}"

            except Exception as e:
                error_msg = f"Attempt {attempt + 1} failed: {str(e)}"
                continue

        duration = time.time() - start_time
        success = is_valid and embeddings is not None

        # Create individual result dictionaries from batch results
        for idx, image_path in enumerate(image_paths):
            embedding = embeddings[idx] if success else None

            result = {
                "embedding": embedding,
                "is_valid": is_valid,
                "success": success,
                "duration": duration / num_images if num_images > 0 else duration,
                "image_path": image_path,
                "model_name": model_name,
                "error": error_msg if not success else None,
            }

            results.append(result)

    except Exception as e:
        # If global error occurs, create error entries for all images
        for image_path in (
            image_paths if isinstance(image_paths, list) else [image_paths]
        ):
            results.append(
                {
                    "embedding": None,
                    "is_valid": False,
                    "success": False,
                    "duration": 0,
                    "image_path": image_path,
                    "model_name": model_name,
                    "error": str(e),
                }
            )

    return results


def vectorize_texts(
    embedder: ClothingImageEmbedder,
    texts: Union[str, List[str]],
    model_name: str = "fashion-clip",
    batch_size: int = 32,
) -> List[dict]:
    """Complete text vectorization pipeline with batch processing and individual results."""
    start_time = time.time()
    results = []

    try:
        # Create embedder
        # embedder = create_embedder(model_name)
        # Handle single text input
        if isinstance(texts, str):
            texts = [texts]

        num_texts = len(texts)
        embeddings = None
        is_valid = False
        error_msg = ""

        try:
            embeddings = embedder.encode_texts(texts, batch_size=batch_size)
            is_valid = embedder.is_valid_embedding(embeddings)

            if not is_valid:
                error_msg = "Invalid embeddings generated"

        except Exception as e:
            error_msg = f"Text encoding failed: {str(e)}"

        duration = time.time() - start_time
        success = is_valid and embeddings is not None

        # Create individual result dictionaries from batch results
        for idx, text in enumerate(texts):
            embedding = embeddings[idx] if success else None

            result = {
                "embedding": embedding,
                "is_valid": is_valid,
                "success": success,
                "duration": duration / num_texts if num_texts > 0 else duration,
                "text": text,
                "model_name": model_name,
                "error": error_msg if not success else None,
            }

            results.append(result)

    except Exception as e:
        # If global error occurs, create error entries for all texts
        for text in texts if isinstance(texts, list) else [texts]:
            results.append(
                {
                    "embedding": None,
                    "is_valid": False,
                    "success": False,
                    "duration": 0,
                    "text": text,
                    "model_name": model_name,
                    "error": str(e),
                }
            )

    return results


def get_embeddings_from_analysis(analysis: dict) -> List[dict]:
    """Extract embeddings from analysis result for database storage."""
    if analysis.get("is_valid") and analysis.get("success"):
        embeddings = {
            "model_name": analysis["model_name"],
            "confidence": model_confidence[analysis["model_name"]],
            "embedding": analysis.get("embedding"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    return embeddings


def get_processing_status(analysis: dict) -> dict:
    """Extract processing status from analysis result for database storage."""
    return {analysis["model_name"]: analysis["is_valid"] and analysis["success"]}
