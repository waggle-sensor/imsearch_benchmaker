# Creating Custom Adapters

The framework supports creating custom adapters for vision annotation, relevance judging, and similarity scoring. Adapters are automatically discovered when placed in the `imsearch_benchmaker/adapters/` directory.

## Adapter Types

There are three types of adapters:

1. **Vision Adapters**: Annotate images with tags, taxonomies, and metadata
2. **Judge Adapters**: Evaluate query-image relevance
3. **Similarity Adapters**: Calculate similarity scores between queries and images

## Step 1: Create Your Adapter Directory

Create a new directory under `imsearch_benchmaker/adapters/`:

```bash
mkdir -p imsearch_benchmaker/adapters/myprovider
```

## Step 2: Implement the Adapter Class

### Vision Adapter Example

```python
# imsearch_benchmaker/adapters/myprovider/vision.py
from typing import Dict, Iterable, List, Any, Optional
from pathlib import Path
from ...framework.vision import Vision
from ...framework.vision_types import VisionImage, VisionAnnotation
from ...framework.config import BenchmarkConfig
from ...framework.cost import CostSummary
from .config import MyProviderVisionConfig

class MyProviderVision(Vision):
    """Custom vision adapter for MyProvider."""
    
    def __init__(self, config: BenchmarkConfig, client: Any = None):
        super().__init__(config, client)
        # Initialize your provider client here
        self.client = client or self._create_client()
    
    def build_request(self, image: VisionImage) -> Dict[str, object]:
        """Build request for a single image."""
        return {
            "image_url": image.image_url,
            "image_id": image.image_id,
            # Add provider-specific fields
        }
    
    def parse_response(self, response_body: Dict[str, object], image: VisionImage) -> VisionAnnotation:
        """Parse provider response to VisionAnnotation."""
        return VisionAnnotation(
            image_id=image.image_id,
            tags=response_body.get("tags", []),
            summary=response_body.get("summary", ""),
            # Map other fields from response_body
        )
    
    def submit(self, images: Iterable[VisionImage], **kwargs) -> object:
        """Submit images and return batch reference."""
        # Implement batch submission logic
        pass
    
    def wait_for_batch(self, batch_ref: object) -> None:
        """Wait for batch to complete."""
        # Implement waiting logic
        pass
    
    def download_batch(self, batch_ref: object, output_path: Path) -> None:
        """Download batch results."""
        # Implement download logic
        pass
    
    def calculate_actual_costs(self, batch_output_jsonl: Path, num_items: Optional[int] = None) -> CostSummary:
        """Calculate actual costs from batch output."""
        # Implement cost calculation
        pass
```

### Judge Adapter Example

```python
# imsearch_benchmaker/adapters/myprovider/judge.py
from typing import Dict, Iterable, Any, Optional
from pathlib import Path
from ...framework.judge import Judge
from ...framework.judge_types import JudgeQuery, JudgeResult
from ...framework.config import BenchmarkConfig
from ...framework.cost import CostSummary
from .config import MyProviderJudgeConfig

class MyProviderJudge(Judge):
    """Custom judge adapter for MyProvider."""
    
    def build_request(self, query: JudgeQuery) -> Dict[str, object]:
        """Build request for a query."""
        return {
            "query_text": query.query_text,
            "query_id": query.query_id,
            "candidate_images": [img.image_url for img in query.candidate_images],
            # Add provider-specific fields
        }
    
    def parse_response(self, response_body: Dict[str, object], query: JudgeQuery) -> JudgeResult:
        """Parse provider response to JudgeResult."""
        return JudgeResult(
            query_id=query.query_id,
            image_id=response_body["image_id"],
            relevance_label=response_body["relevance"],
            confidence=response_body.get("confidence"),
        )
    
    def submit(self, queries: Iterable[JudgeQuery], **kwargs) -> object:
        """Submit queries and return batch reference."""
        # Implement batch submission logic
        pass
    
    def wait_for_batch(self, batch_ref: object) -> None:
        """Wait for batch to complete."""
        # Implement waiting logic
        pass
    
    def download_batch(self, batch_ref: object, output_path: Path) -> None:
        """Download batch results."""
        # Implement download logic
        pass
    
    def calculate_actual_costs(self, batch_output_jsonl: Path, num_items: Optional[int] = None) -> CostSummary:
        """Calculate actual costs from batch output."""
        # Implement cost calculation
        pass
```

### Similarity Adapter Example

```python
# imsearch_benchmaker/adapters/myprovider/similarity.py
from typing import Iterable, List, Optional
from ...framework.scoring import Similarity
from ...framework.scoring_types import SimilarityInput, SimilarityResult
from ...framework.config import BenchmarkConfig
from .config import MyProviderSimilarityConfig

class MyProviderSimilarity(Similarity):
    """Custom similarity adapter for MyProvider."""
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config
        # Initialize your model/service here
    
    def score(self, query: str, image_url: str) -> float:
        """Calculate similarity score for a query-image pair."""
        # Implement scoring logic
        return 0.0
    
    def score_batch(self, inputs: Iterable[SimilarityInput]) -> List[SimilarityResult]:
        """Calculate similarity scores for multiple pairs."""
        results = []
        for inp in inputs:
            score = self.score(inp.query, inp.image_url)
            results.append(SimilarityResult(
                query_id=inp.query_id,
                image_id=inp.image_id,
                score=score
            ))
        return results
```

## Step 3: Create Configuration Class

If needed, you can create a config class that extends the base config class for the adapter:

>NOTE: Not all adapters need a new adapter config class. It is only needed if you need to add new configuration parameters to the adapter.

```python
# imsearch_benchmaker/adapters/myprovider/config.py
from dataclasses import dataclass
from typing import Optional
from ...framework.config import VisionConfig, JudgeConfig, SimilarityConfig

@dataclass(frozen=True)
class MyProviderVisionConfig(VisionConfig):
    """Configuration for MyProvider vision adapter."""
    model: Optional[str] = "my-model-v1"
    api_key: Optional[str] = None  # Use _ prefix for sensitive fields
    # Add provider-specific config fields

@dataclass(frozen=True)
class MyProviderJudgeConfig(JudgeConfig):
    """Configuration for MyProvider judge adapter."""
    model: Optional[str] = "my-model-v1"
    # Add provider-specific config fields

@dataclass(frozen=True)
class MyProviderSimilarityConfig(SimilarityConfig):
    """Configuration for MyProvider similarity adapter."""
    endpoint: Optional[str] = None
    # Add provider-specific config fields
```

## Step 4: Register Your Adapter

Update `imsearch_benchmaker/adapters/__init__.py` to register your adapter:

```python
# imsearch_benchmaker/adapters/__init__.py
from ..framework.vision import VisionAdapterRegistry
from ..framework.judge import JudgeAdapterRegistry
from ..framework.scoring import SimilarityAdapterRegistry

# Import your adapters
from .myprovider import (
    MyProviderVision,
    MyProviderJudge,
    MyProviderSimilarity,
    MyProviderVisionConfig,
    MyProviderJudgeConfig,
    MyProviderSimilarityConfig,
)

# Register vision adapter
VisionAdapterRegistry.register(
    "myprovider",
    MyProviderVision,
    config_class=MyProviderVisionConfig
)

# Register judge adapter
JudgeAdapterRegistry.register(
    "myprovider",
    MyProviderJudge,
    config_class=MyProviderJudgeConfig
)

# Register similarity adapter
SimilarityAdapterRegistry.register(
    "myprovider",
    MyProviderSimilarity,
    config_class=MyProviderSimilarityConfig
)
```

## Step 5: Use Your Adapter

Update your `config.toml`:

```toml
[vision_config]
adapter = "myprovider"
model = "my-model-v1"

[judge_config]
adapter = "myprovider"
model = "my-model-v1"

[similarity_config]
adapter = "myprovider"
endpoint = "https://api.myprovider.com/similarity"
```

## Adapter Requirements

### Vision Adapter Must Implement:

- `build_request(image: VisionImage) -> Dict`: Convert image to provider request
- `parse_response(response: Dict, image: VisionImage) -> VisionAnnotation`: Parse provider response
- `submit(images: Iterable[VisionImage]) -> object`: Submit batch and return reference
- `wait_for_batch(batch_ref: object) -> None`: Wait for batch completion
- `download_batch(batch_ref: object, output_path: Path) -> None`: Download results
- `calculate_actual_costs(batch_output_jsonl: Path) -> CostSummary`: Calculate costs

### Judge Adapter Must Implement:

- `build_request(query: JudgeQuery) -> Dict`: Convert query to provider request
- `parse_response(response: Dict, query: JudgeQuery) -> JudgeResult`: Parse provider response
- `submit(queries: Iterable[JudgeQuery]) -> object`: Submit batch and return reference
- `wait_for_batch(batch_ref: object) -> None`: Wait for batch completion
- `download_batch(batch_ref: object, output_path: Path) -> None`: Download results
- `calculate_actual_costs(batch_output_jsonl: Path) -> CostSummary`: Calculate costs

### Similarity Adapter Must Implement:

- `score(query: str, image_url: str) -> float`: Calculate single similarity score
- `score_batch(inputs: Iterable[SimilarityInput]) -> List[SimilarityResult]`: Batch scoring

## Best Practices

1. **Error Handling**: Implement robust error handling for API failures
2. **Cost Tracking**: Implement `calculate_actual_costs()` for accurate cost monitoring
3. **Batch Support**: For vision/judge adapters, implement efficient batch processing
4. **Configuration**: Use dataclasses with `frozen=True` for config classes
5. **Sensitive Data**: Prefix sensitive fields with `_` (e.g., `_api_key`) and load from environment
6. **Type Hints**: Use proper type hints for better IDE support
7. **Documentation**: Add docstrings explaining adapter-specific behavior

## Example: Complete Adapter Structure

This example shows an adapter providing vision annotation, relevance judging, and similarity scoring, but you can create adapters with any combination of these or only one of them.

```
imsearch_benchmaker/
└── adapters/
    └── myprovider/
        ├── __init__.py          # Exports adapter classes
        ├── config.py            # Config classes 
        ├── vision.py            # Vision adapter 
        ├── judge.py             # Judge adapter 
        ├── similarity.py        # Similarity adapter
        └── requirements.txt     # Adapter-specific dependencies
```

## Testing Your Adapter

Test your adapter by running the pipeline:

```bash
benchmaker vision --adapter myprovider --config config.toml
benchmaker judge --adapter myprovider --config config.toml
benchmaker postprocess similarity --config config.toml
```

## Reference Implementations

For more examples, see the existing adapters:

- **OpenAI Adapters**: `imsearch_benchmaker/adapters/openai/` - Complete implementation with batch processing, cost tracking, and error handling
- **Local CLIP Adapter**: `imsearch_benchmaker/adapters/local/` - Simple similarity adapter example

These reference implementations demonstrate best practices and can serve as templates for your own adapters.

