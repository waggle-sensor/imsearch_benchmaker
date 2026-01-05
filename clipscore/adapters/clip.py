"""
clip.py

CLIP API adapters for different services.

To add a new adapter, subclass the CLIPAdapter class and implement the score and get_name methods.
Then register the adapter with the CLIPAdapterRegistry.

Example:
class MyCLIPAdapter(CLIPAdapter):
    def score(self, query: str, image_url: str) -> float:
        return 0.0
    def get_name(self) -> str:
        return "my_clip_adapter"
CLIPAdapterRegistry.register("my_clip_adapter", MyCLIPAdapter)
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional
import requests
import torch
from ..framework.base import CLIPAdapter

class HTTPCLIPAdapter(CLIPAdapter):
    """
    Generic HTTP-based CLIP adapter that works with REST APIs.
    Can be configured for different services via environment variables.
    """
    
    def __init__(
        self,
        api_url: str,
        api_key: Optional[str] = None,
        api_key_header: str = "Authorization",
        api_key_format: str = "Bearer {key}",
        request_timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize the HTTP CLIP adapter.
        
        Args:
            api_url: The base URL of the CLIP API endpoint.
            api_key: Optional API key for authentication.
            api_key_header: HTTP header name for the API key.
            api_key_format: Format string for the API key (use {key} placeholder).
            request_timeout: Request timeout in seconds.
            max_retries: Maximum number of retry attempts.
            retry_delay: Delay between retries in seconds.
        """
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.api_key_header = api_key_header
        self.api_key_format = api_key_format
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    def _make_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make an HTTP request to the CLIP API with retry logic.
        
        Args:
            payload: The request payload.
            
        Returns:
            The JSON response as a dictionary.
            
        Raises:
            Exception: If the request fails after all retries.
        """
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers[self.api_key_header] = self.api_key_format.format(key=self.api_key)
        
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    json=payload,
                    headers=headers,
                    timeout=self.request_timeout
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise Exception(f"CLIP API request failed after {self.max_retries} attempts: {e}") from e
        
        raise Exception(f"CLIP API request failed: {last_error}") from last_error
    
    def _build_payload(self, query: str, image_url: str) -> Dict[str, Any]:
        """
        Build the request payload for the CLIP API.
        Override this method in subclasses to customize the payload format.
        
        Args:
            query: The text query.
            image_url: The image URL.
            
        Returns:
            The request payload dictionary.
        """
        return {
            "text": query,
            "image_url": image_url
        }
    
    def _extract_score(self, response: Dict[str, Any]) -> float:
        """
        Extract the CLIPScore from the API response.
        Override this method in subclasses to customize response parsing.
        
        Args:
            response: The JSON response from the API.
            
        Returns:
            The CLIPScore as a float.
        """
        # Try common response field names
        for field in ["score", "clip_score", "similarity", "similarity_score", "result"]:
            if field in response:
                value = response[field]
                if isinstance(value, (int, float)):
                    return float(value)
                elif isinstance(value, dict) and "score" in value:
                    return float(value["score"])
        
        # If no standard field found, try to find any numeric value
        for key, value in response.items():
            if isinstance(value, (int, float)):
                return float(value)
        
        raise ValueError(f"Could not extract score from response: {response}")
    
    def score(self, query: str, image_url: str) -> float:
        """
        Calculate CLIPScore for a query-image pair.
        
        Args:
            query: The text query.
            image_url: The URL of the image.
            
        Returns:
            A float representing the CLIPScore.
        """
        payload = self._build_payload(query, image_url)
        response = self._make_request(payload)
        return self._extract_score(response)
    
    def get_name(self) -> str:
        """Get the name of this adapter."""
        return "HTTP_CLIP"

class LocalCLIPAdapter(CLIPAdapter):
    """
    Adapter for locally hosted CLIP models using Hugging Face transformers.
    
    This adapter loads CLIP models directly from Hugging Face and runs inference locally.
    Supports both CPU and GPU execution.
    """
    
    def __init__(
        self,
        model: str = "openai/clip-vit-base-patch32",
        device: str = "auto",
        torch_dtype: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the local CLIP adapter.
        
        Args:
            model: The Hugging Face model identifier (e.g., "openai/clip-vit-base-patch32").
            device: Device to run inference on. Options: "auto", "cpu", "cuda", "cuda:0", etc.
                   "auto" will use CUDA if available, otherwise CPU.
            torch_dtype: Optional torch dtype for the model (e.g., "float16", "bfloat16").
                        If None, uses the model's default dtype.
            **kwargs: Additional arguments.
        """
        try:
            from transformers import CLIPModel, AutoProcessor
        except ImportError:
            raise ImportError(
                "transformers library is required for LocalCLIPAdapter. "
                "Install it with: pip install transformers torch torchvision"
            )
        
        self.model_name = model
        self.device = self._determine_device(device)
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(model)
        
        # Load model with optional dtype
        model_kwargs = {**kwargs}
        if torch_dtype:
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            if torch_dtype.lower() in dtype_map:
                model_kwargs["torch_dtype"] = dtype_map[torch_dtype.lower()]
            else:
                raise ValueError(f"Unsupported torch_dtype: {torch_dtype}. Use one of: {list(dtype_map.keys())}")
        
        self.model = CLIPModel.from_pretrained(model, **model_kwargs)
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
    
    def get_name(self) -> str:
        """Get the name of this adapter."""
        return f"Local_CLIP_{self.model_name}_{self.device.type}"
