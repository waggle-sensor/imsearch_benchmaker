# OpenAI Adapters

OpenAI adapters for vision annotation and relevance judging using OpenAI's Batch API and structured outputs.

## Overview

The OpenAI adapters provide:
- **Vision Adapter** (`OpenAIVision`): Annotates images with structured outputs including tags, taxonomies, boolean fields, summaries, and confidence scores
- **Judge Adapter** (`OpenAIJudge`): Evaluates query-image relevance with binary or graded relevance labels
- **Vision Adapter Configuration** (`OpenAIVisionConfig`): Configuration for the vision adapter. It extends the [VisionConfig](../../framework/config.py#L146) class to add OpenAI-specific configuration.
- **Judge Adapter Configuration** (`OpenAIJudgeConfig`): Configuration for the judge adapter. It extends the [JudgeConfig](../../framework/config.py#L146) class to add OpenAI-specific configuration.

Both adapters use OpenAI's Batch API for efficient processing of large datasets and support structured outputs via JSON schema.

## Installation

Install the OpenAI adapters:

```bash
pip install imsearch-benchmaker[openai]
```

Or install all adapters:

```bash
pip install imsearch-benchmaker[all]
```

## Dependencies

see [requirements.txt](requirements.txt) for more details.

## Configuration

### Vision Adapter Configuration

Configure the OpenAI vision adapter in your `config.toml`:

```toml
[vision_config]
adapter = "openai"
model = "gpt-4o"  # or "gpt-4o-mini", "gpt-4-turbo", etc.
system_prompt = "You are an expert image annotator..."
user_prompt = "Annotate this image with the following fields..."
max_output_tokens = 4000
reasoning_effort = "medium"  # "low", "medium", "high" (for o1 models)
controlled_tag_vocab = ["tag1", "tag2", ...]  # Optional: controlled vocabulary
min_tags = 12  # Minimum tags per image
max_tags = 18  # Maximum tags per image

# Cost tracking (required for cost calculation)
price_per_million_input_tokens = 2.50
price_per_million_output_tokens = 10.00
price_per_million_image_input_tokens = 2.50
price_per_million_image_output_tokens = 10.00
price_per_million_cached_input_tokens = 0.25  # Optional: for prompt caching
```

### Judge Adapter Configuration

Configure the OpenAI judge adapter in your `config.toml`:

```toml
[judge_config]
adapter = "openai"
model = "gpt-4o"
system_prompt = "You are an expert at evaluating image search relevance..."
user_prompt = "Evaluate the relevance of these images for the query..."
max_output_tokens = 8000
reasoning_effort = "medium"  # "low", "medium", "high" (for o1 models)

# Cost tracking (required for cost calculation)
price_per_million_input_tokens = 2.50
price_per_million_output_tokens = 10.00
price_per_million_cached_input_tokens = 0.25  # Optional: for prompt caching
```

### Sensitive Fields

Set the OpenAI API key via environment variable:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or in config (not recommended for production):

```toml
[vision_config]
_openai_api_key = "your-api-key-here"  # Fields starting with _ are sensitive
```

## Features

### Vision Adapter Features

- **Structured Outputs**: Uses JSON schema to enforce structured responses
- **Dynamic Schema Generation**: Automatically builds JSON schema from `BenchmarkConfig`:
  - Summary field (always included)
  - Taxonomy columns (enum values from `config.columns_taxonomy`)
  - Boolean columns (from `config.columns_boolean`)
  - Tags (controlled vocabulary from `config.vision_config.controlled_tag_vocab`)
  - Confidence scores (for taxonomy and boolean columns)
- **Batch Processing**: Uses OpenAI Batch API for efficient processing
- **Automatic Sharding**: Large batches are automatically split into shards
- **Cost Tracking**: Tracks token usage and calculates costs
- **Error Handling**: Retry mechanisms for failed requests
- **Progress Tracking**: Progress bars for batch operations

### Judge Adapter Features

- **Structured Outputs**: Uses JSON schema for consistent relevance judgments
- **Query Generation**: Generates query text from seed images
- **Relevance Evaluation**: Binary (0/1) or graded relevance labels
- **Batch Processing**: Uses OpenAI Batch API for efficient processing
- **Automatic Sharding**: Large batches are automatically split into shards
- **Cost Tracking**: Tracks token usage and calculates costs
- **Error Handling**: Retry mechanisms for failed requests
- **Progress Tracking**: Progress bars for batch operations

## Programmatic Usage

### Vision Adapter

```python
from imsearch_benchmaker.framework.config import BenchmarkConfig
from imsearch_benchmaker.adapters.openai import OpenAIVision
from imsearch_benchmaker.framework.vision_types import VisionImage

config = BenchmarkConfig.from_file("config.toml")
adapter = OpenAIVision(config=config)

# Build request for an image
image = VisionImage(
    image_id="img1",
    image_url="https://example.com/image.jpg"
)
request = adapter.build_request(image)

# Submit batch
batch_ref = adapter.submit([image1, image2, ...])
adapter.wait_for_batch(batch_ref)
adapter.download_batch(batch_ref, Path("output.jsonl"))
```

### Judge Adapter

```python
from imsearch_benchmaker.framework.config import BenchmarkConfig
from imsearch_benchmaker.adapters.openai import OpenAIJudge
from imsearch_benchmaker.framework.judge_types import JudgeQuery

config = BenchmarkConfig.from_file("config.toml")
adapter = OpenAIJudge(config=config)

# Build request for a query
query = JudgeQuery(
    query_id="q001",
    seed_images=[...],
    candidate_images=[...]
)
request = adapter.build_request(query)

# Submit batch
batch_ref = adapter.submit([query1, query2, ...])
adapter.wait_for_batch(batch_ref)
adapter.download_batch(batch_ref, Path("output.jsonl"))
```

## JSON Schema Generation

### Vision Schema

The vision adapter automatically generates a JSON schema based on your `BenchmarkConfig`:

- **Summary**: Always included as a string field
- **Taxonomy Columns**: Each column in `config.columns_taxonomy` becomes an enum field
- **Boolean Columns**: Each column in `config.columns_boolean` becomes a boolean field
- **Tags**: If `config.vision_config.controlled_tag_vocab` is set, tags become an array of enum values
- **Confidence**: Object containing confidence scores (0-1) for taxonomy and boolean columns

Example schema structure:
```json
{
  "type": "object",
  "properties": {
    "summary": {"type": "string"},
    "environment_type": {"type": "string", "enum": ["mountainous", "forest", ...]},
    "flame_visible": {"type": "boolean"},
    "tags": {
      "type": "array",
      "items": {"type": "string", "enum": ["tag1", "tag2", ...]},
      "minItems": 12,
      "maxItems": 18
    },
    "confidence": {
      "type": "object",
      "properties": {
        "environment_type": {"type": "number", "minimum": 0, "maximum": 1},
        "flame_visible": {"type": "number", "minimum": 0, "maximum": 1}
      }
    }
  }
}
```

### Judge Schema

The judge adapter generates a schema for query text and judgments:

```json
{
  "type": "object",
  "properties": {
    "query_text": {"type": "string"},
    "judgments": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "image_id": {"type": "string"},
          "relevance_label": {"type": "integer", "enum": [0, 1]}
        }
      }
    }
  }
}
```

## Batch Processing

### Automatic Sharding

Large batches are automatically split into shards to meet OpenAI's batch size limits. The adapter handles:
- Shard creation
- Parallel submission
- Status tracking
- Result aggregation

### Batch Lifecycle

1. **Make**: Create batch input JSONL file
2. **Submit**: Submit batch(es) to OpenAI
3. **Wait**: Poll for completion (with progress bars)
4. **Download**: Download results
5. **Parse**: Parse results into framework format

### Batch Status

Batches can be in the following states:
- `validating`: Batch is being validated
- `in_progress`: Batch is processing
- `finalizing`: Batch is finalizing
- `completed`: Batch completed successfully
- `failed`: Batch failed
- `expired`: Batch expired
- `cancelling`: Batch is being cancelled
- `cancelled`: Batch was cancelled

## Cost Tracking

The adapters track costs based on:
- Input tokens (text)
- Output tokens (text)
- Image input tokens (for vision)
- Image output tokens (for vision)
- Cached input tokens (optional, for prompt caching)

Cost summaries are generated and can be exported to CSV. Set pricing in your config:

```toml
[vision_config]
price_per_million_input_tokens = 2.50
price_per_million_output_tokens = 10.00
price_per_million_image_input_tokens = 2.50
price_per_million_image_output_tokens = 10.00
```

## Error Handling

### Retry Failed Requests

If some requests fail, you can retry them:

```bash
benchmaker vision-retry --config config.toml --submit
benchmaker judge-retry --config config.toml --submit
```

### Error Files

Failed requests are saved to error JSONL files:
- `vision_batch_error.jsonl`
- `judge_batch_error.jsonl`

These files contain the original request and error information for debugging.

## List Batches

View all batches:

```bash
benchmaker list-batches --config config.toml
benchmaker list-batches --active-only --config config.toml
```

## Supported Models

### Vision Models
- `gpt-4o` (recommended)
- `gpt-4o-mini`
- `gpt-4-turbo`
- `gpt-4-vision-preview`

### Judge Models
- `gpt-4o` (recommended)
- `gpt-4o-mini`
- `gpt-4-turbo`
- `o1-preview` (with reasoning_effort)
- `o1-mini` (with reasoning_effort)

## Best Practices

1. **Use Batch API**: Always use batch processing for large datasets (more cost-effective)
2. **Set Pricing**: Configure pricing for accurate cost tracking
3. **Monitor Costs**: Review cost summaries in the `summary/` directory
4. **Handle Errors**: Use retry commands for failed requests
5. **Use Controlled Vocabularies**: For tags, use controlled vocabularies for consistency
6. **Optimize Prompts**: Well-crafted prompts improve annotation quality
7. **Test with Small Batches**: Test your configuration with small batches first

## Troubleshooting

### Authentication Errors

Ensure `OPENAI_API_KEY` is set:
```bash
export OPENAI_API_KEY="your-key-here"
```

### Cost Calculation Errors

Ensure all required pricing fields are set in config:
- `price_per_million_input_tokens`
- `price_per_million_output_tokens`
- For vision: `price_per_million_image_input_tokens` and `price_per_million_image_output_tokens`

### Batch Timeout

Batches have a completion window (default: 24 hours). If a batch times out, check:
- Batch size (may need to be smaller)
- Model availability
- API status

### Schema Validation Errors

If you see schema validation errors:
- Check that your config matches the expected schema
- Verify enum values in `columns_taxonomy` are correct
- Ensure `columns_boolean` list is correct

## API Reference

### OpenAIVision

- `build_json_schema(config: BenchmarkConfig) -> Dict[str, Any]`: Build JSON schema from config
- `build_request(image: VisionImage) -> Dict[str, object]`: Build OpenAI request
- `parse_response(response_body: Dict[str, object], image: VisionImage) -> VisionAnnotation`: Parse response
- `submit(images: Iterable[VisionImage]) -> BatchRefs`: Submit batch
- `wait_for_batch(batch_ref: BatchRefs) -> None`: Wait for completion
- `download_batch(batch_ref: BatchRefs, output_path: Path) -> None`: Download results
- `calculate_actual_costs(batch_output_jsonl: Path, num_items: Optional[int]) -> CostSummary`: Calculate costs

### OpenAIJudge

- `build_json_schema(config: BenchmarkConfig) -> Dict[str, Any]`: Build JSON schema from config
- `build_request(query: JudgeQuery) -> Dict[str, object]`: Build OpenAI request
- `parse_response(response_body: Dict[str, object], query: JudgeQuery) -> JudgeResult`: Parse response
- `submit(queries: Iterable[JudgeQuery]) -> BatchRefs`: Submit batch
- `wait_for_batch(batch_ref: BatchRefs) -> None`: Wait for completion
- `download_batch(batch_ref: BatchRefs, output_path: Path) -> None`: Download results
- `calculate_actual_costs(batch_output_jsonl: Path, num_items: Optional[int]) -> CostSummary`: Calculate costs

## Files

- `vision.py`: Vision adapter implementation
- `judge.py`: Judge adapter implementation
- `config.py`: Configuration classes (`OpenAIVisionConfig`, `OpenAIJudgeConfig`, `OpenAIConfig`)
- `batch.py`: Batch API utilities (sharding, submission, polling)
- `client.py`: OpenAI client initialization
- `_responses.py`: Response parsing and cost calculation utilities
- `requirements.txt`: Dependencies (`openai>=2.13.0`)

