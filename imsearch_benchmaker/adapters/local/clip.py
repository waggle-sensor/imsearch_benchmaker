"""
clip.py

Local CLIP adapter for similarity scoring using Hugging Face transformers.
"""

from __future__ import annotations

from typing import Iterable, List, Optional
import torch

from ...framework.scoring import Similarity, SimilarityAdapterRegistry
from ...framework.scoring_types import SimilarityInput, SimilarityResult
from ...framework.config import BenchmarkConfig, DEFAULT_BENCHMARK_CONFIG
from .config import CLIPConfig


class CLIP(Similarity):
    """
    Adapter for locally hosted CLIP models using Hugging Face transformers.
    
    This adapter loads CLIP models directly from Hugging Face and runs inference locally.
    Supports both CPU and GPU execution.
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        device: Optional[str] = None,
        torch_dtype: Optional[str] = None,
        use_safetensors: Optional[bool] = None,
        config: Optional[BenchmarkConfig] = None,
    ) -> None:
        """
        Initialize the local CLIP adapter.
        
        Args:
            model: The Hugging Face model identifier (e.g., "openai/clip-vit-base-patch32").
            device: Device to run inference on. Options: "auto", "cpu", "cuda", "cuda:0", etc.
                   "auto" will use CUDA if available, otherwise CPU.
            torch_dtype: Optional torch dtype for the model (e.g., "float16", "bfloat16").
                        If None, uses the model's default dtype.
            use_safetensors: Whether to use safetensors format for model loading.
            config: Optional BenchmarkConfig instance.
        """
        try:
            from transformers import CLIPModel, AutoProcessor
        except ImportError:
            raise ImportError(
                "transformers library is required for local CLIP adapter. "
                "Install it with: pip install transformers torch torchvision"
            )
        if not isinstance(self.config.similarity_config, CLIPConfig):
            raise ValueError("local CLIP adapter requires CLIPConfig in config.similarity_config.")
        if not self.model:
            raise ValueError("local CLIP adapter requires a model name.")
        
        config = config or DEFAULT_BENCHMARK_CONFIG
        self.config = config
        similarity_cfg = config.similarity_config
        
        # Get values from config or parameters
        self.model_name = model or similarity_cfg.model or "openai/clip-vit-base-patch32"
        
        # Get CLIP-specific config if available
        self.device_str = device or similarity_cfg.device or "auto"
        self.torch_dtype_str = torch_dtype or similarity_cfg.torch_dtype
        self.use_safetensors = use_safetensors if use_safetensors is not None else (similarity_cfg.use_safetensors if similarity_cfg.use_safetensors is not None else True)
        
        self.device = self._determine_device(self.device_str)
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        
        # Load model with optional dtype
        model_kwargs = {}
        if self.torch_dtype_str:
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            if self.torch_dtype_str.lower() in dtype_map:
                model_kwargs["torch_dtype"] = dtype_map[self.torch_dtype_str.lower()]
            else:
                raise ValueError(f"Unsupported torch_dtype: {self.torch_dtype_str}. Use one of: {list(dtype_map.keys())}")
        
        # Add safetensors parameter if supported
        if self.use_safetensors:
            try:
                model_kwargs["use_safetensors"] = True
            except TypeError:
                # Older versions of transformers don't support use_safetensors parameter
                pass
        
        self.model = CLIPModel.from_pretrained(self.model_name, **model_kwargs)
        self.model.to(self.device)
        self.model.eval()
    
    def _determine_device(self, device: str) -> torch.device:
        """
        Determine the device to use for inference.
        
        Args:
            device: Device specification string.
            
        Returns:
            torch.device object.
        """
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device)
    
    def score(self, query: str, image_url: str) -> float:
        """
        Calculate similarity score between the query and image.
        
        Args:
            query: The text query.
            image_url: The URL of the image.
            
        Returns:
            A float representing the similarity score between the query and image.
        """
        # Load and process image
        image = self._load_image_from_url(image_url)
        
        # Process inputs
        inputs = self.processor(
            text=[query],
            images=image,
            return_tensors="pt",
            padding=True
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # For a single query-image pair, get the similarity score
            # CLIP uses an image encoder and text encoder to get visual features and text features. 
            # Both features are projected to a latent space with the same number of dimensions and their dot product gives a similarity score.
            logits_per_image = outputs.logits_per_image
            similarity = logits_per_image[0][0]
        
        return float(similarity)
    
    def score_batch(self, inputs: Iterable[SimilarityInput]) -> List[SimilarityResult]:
        """
        Calculate similarity scores for multiple query-image pairs.
        
        Args:
            inputs: Iterable of SimilarityInput objects.
            
        Returns:
            List of SimilarityResult objects.
        """
        results = []
        for input_item in inputs:
            score = self.score(input_item.query, input_item.image_url)
            results.append(SimilarityResult(
                query=input_item.query,
                image_url=input_item.image_url,
                score=score,
                query_id=input_item.query_id,
                image_id=input_item.image_id,
            ))
        return results
    
    def get_name(self) -> str:
        """Get the name of this adapter."""
        return f"local_clip_{self.model_name.replace('/', '_')}_{self.device.type}"


# Register the adapter
SimilarityAdapterRegistry.register("local_clip", CLIP, config_class=CLIPConfig)

