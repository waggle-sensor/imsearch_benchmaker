# Local Adapter

Adapter for using local models using Hugging Face transformers. This adapter runs models locally, eliminating API costs and enabling offline processing.

>NOTE: The adapter only has a similarity scoring feature for now. A Judge and Vision adapter can be created to support local running of these models.

## Overview

The Local CLIP adapter (`CLIP`) calculates similarity scores between text queries and images using CLIP (Contrastive Language-Image Pre-training) models. It supports any CLIP model available on Hugging Face and can run on both CPU and GPU.

## Installation

Install the local adapter:

```bash
pip install imsearch-benchmaker[local]
```

Or install all adapters:

```bash
pip install imsearch-benchmaker[all]
```

## Dependencies

see [requirements.txt](requirements.txt) for more details.

## Configuration

Configure the local CLIP adapter in your `config.toml`:

```toml
[similarity_config]
adapter = "local_clip"
model = "openai/clip-vit-base-patch32"  # Hugging Face model ID
device = "auto"  # "auto", "cpu", "cuda", "cuda:0", etc.
torch_dtype = "float32"  # Optional: "float16", "bfloat16", "float32"
use_safetensors = true  # Use safetensors format
col_name = "clip_score"  # Column name for similarity score
```

### Configuration Options

- **model** (str): Hugging Face model identifier. Examples:
  - `openai/clip-vit-base-patch32` (default)
  - `openai/clip-vit-large-patch14`
  - `laion/CLIP-ViT-B-32-xlm-roberta-base-laion5B-s13B-b90k`
  - Any CLIP model on Hugging Face Hub

- **device** (str): Device to run inference on
  - `"auto"` (default): Automatically use CUDA if available, otherwise CPU
  - `"cpu"`: Force CPU execution
  - `"cuda"`: Use default CUDA device
  - `"cuda:0"`, `"cuda:1"`, etc.: Use specific CUDA device

- **torch_dtype** (str, optional): Torch data type for model
  - `"float32"`: Full precision (default)
  - `"float16"`: Half precision (faster, less memory, slight accuracy loss)
  - `"bfloat16"`: Brain float (better numerical stability than float16)

- **use_safetensors** (bool): Whether to use safetensors format (default: `true`)

- **col_name** (str): Column name for similarity score in output (default: `"clip_score"`)

## Usage

### CLI Usage

```bash
# Calculate similarity scores
benchmaker postprocess similarity --config config.toml
```

### Programmatic Usage

```python
from imsearch_benchmaker.framework.config import BenchmarkConfig
from imsearch_benchmaker.adapters.local import CLIP
from imsearch_benchmaker.framework.scoring_types import SimilarityInput

config = BenchmarkConfig.from_file("config.toml")
adapter = CLIP(config=config)

# Calculate single similarity score
score = adapter.score(
    query="a red car on a highway",
    image_url="https://example.com/image.jpg"
)

# Calculate batch similarity scores
inputs = [
    SimilarityInput(
        query="a red car",
        image_url="https://example.com/car.jpg",
        query_id="q001",
        image_id="img001"
    ),
    # ... more inputs
]
results = adapter.score_batch(inputs)
```

## Features

- **No API Costs**: Runs entirely locally, no API calls
- **GPU Support**: Automatic GPU detection and usage
- **Any CLIP Model**: Supports any CLIP model from Hugging Face
- **Batch Processing**: Efficient batch processing with progress tracking
- **Flexible Precision**: Support for float32, float16, and bfloat16
- **Image URL Loading**: Automatically downloads and processes images from URLs
- **Error Handling**: Robust error handling for image loading failures

## Supported Models

Any CLIP model from Hugging Face Hub is supported. Popular options:

- **OpenAI CLIP Models**:
  - `openai/clip-vit-base-patch32` (default, 151M parameters)
  - `openai/clip-vit-base-patch16` (86M parameters)
  - `openai/clip-vit-large-patch14` (427M parameters)

- **LAION CLIP Models**:
  - `laion/CLIP-ViT-B-32-xlm-roberta-base-laion5B-s13B-b90k`
  - `laion/CLIP-ViT-L-14-laion2B-s32B-b82K`

- **Multilingual CLIP Models**:
  - `sentence-transformers/clip-ViT-B-32-multilingual-v1`

## Performance

### GPU vs CPU

- **GPU**: Significantly faster (10-100x speedup depending on model)
- **CPU**: Slower but works on any machine

### Model Size vs Speed

- **Smaller models** (e.g., `clip-vit-base-patch32`): Faster inference, lower accuracy
- **Larger models** (e.g., `clip-vit-large-patch14`): Slower inference, higher accuracy

### Precision Trade-offs

- **float32**: Best accuracy, slower, more memory
- **float16**: Good accuracy, faster, less memory (recommended for GPU)
- **bfloat16**: Good accuracy, faster, less memory, better numerical stability

## Best Practices

1. **Use GPU When Available**: Set `device = "auto"` to automatically use GPU
2. **Choose Appropriate Model**: Balance speed vs accuracy for your use case
3. **Use float16 on GPU**: Reduces memory usage and increases speed with minimal accuracy loss
4. **Batch Processing**: Process multiple query-image pairs together for efficiency
5. **Image Caching**: Consider caching downloaded images for repeated processing

## Troubleshooting

### CUDA Out of Memory

If you run out of GPU memory:
- Use a smaller model
- Use `torch_dtype = "float16"` or `"bfloat16"`
- Process smaller batches
- Use CPU instead: `device = "cpu"`

### Import Errors

If you see import errors:
```bash
pip install transformers torch torchvision
```

### Slow Performance

- Ensure GPU is being used: Check logs for device information
- Use a smaller model if accuracy requirements allow
- Use `float16` or `bfloat16` for faster inference
- Process in larger batches

### Image Loading Errors

If images fail to load:
- Check image URLs are accessible
- Verify network connectivity
- Check image format is supported (JPEG, PNG, etc.)
- Review error messages in logs

## API Reference

### CLIP

- `__init__(model: Optional[str] = None, device: Optional[str] = None, torch_dtype: Optional[str] = None, use_safetensors: Optional[bool] = None, config: Optional[BenchmarkConfig] = None)`: Initialize adapter
- `score(query: str, image_url: str) -> float`: Calculate similarity score for a single query-image pair
- `score_batch(inputs: Iterable[SimilarityInput]) -> List[SimilarityResult]`: Calculate similarity scores for multiple pairs
- `_load_image_from_url(image_url: str) -> Image.Image`: Load image from URL (helper method)

## Files

- `clip.py`: CLIP adapter implementation
- `config.py`: Configuration class (`CLIPConfig`)
- `requirements.txt`: Dependencies (`transformers>=4.57.0`, `torch>=2.2.0`, `torchvision>=0.17.0`)

## Example Configuration

```toml
[similarity_config]
adapter = "local_clip"
model = "openai/clip-vit-base-patch32"
device = "auto"
torch_dtype = "float16"  # Use half precision for faster GPU inference
use_safetensors = true
col_name = "clip_score"
```
