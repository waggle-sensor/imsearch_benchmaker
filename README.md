# Image Search Benchmark Maker

A comprehensive framework for creating structured image search evaluation datasets. This tool automates the entire pipeline from image preprocessing to dataset upload, making it easy to build high-quality benchmarks for image retrieval systems.

## Features

- **Complete Pipeline Automation**: End-to-end workflow from raw images to published datasets
- **Flexible Adapter System**: Pluggable adapters for vision annotation, relevance judging, and similarity scoring
- **Batch Processing**: Efficient batch processing for large-scale datasets.
- **Query Planning**: Intelligent query generation with diversity and difficulty balancing
- **Comprehensive Analysis**: Automatic generation of dataset summaries, statistics, and visualizations
- **Hugging Face Integration**: Direct upload to Hugging Face Hub with dataset cards and summaries
- **Cost Tracking**: Monitor API costs throughout the pipeline
- **Progress Tracking**: Built-in progress bars and logging for long-running operations

## Installation

### Basic Installation

```bash
pip install imsearch-benchmaker
```

### With Optional Adapters

Install with specific adapters:

```bash
# OpenAI adapters (for vision annotation and relevance judging)
pip install imsearch-benchmaker[openai]

# Local CLIP adapter (for similarity scoring)
pip install imsearch-benchmaker[local]

# All adapters
pip install imsearch-benchmaker[all]
```

### Development Installation

```bash
git clone https://github.com/waggle-sensor/imsearch_benchmaker.git
cd imsearch_benchmaker
pip install -e .
```

## Quick Start

### 1. Create a Configuration File

Create a `config.toml` file with your benchmark settings:

```toml
# Basic configuration
benchmark_name = "MyBenchmark"
benchmark_description = "A benchmark for image search"
log_level = "INFO"

# File paths
image_root_dir = "/path/to/images"
images_jsonl = "outputs/images.jsonl"
annotations_jsonl = "outputs/annotations.jsonl"
query_plan_jsonl = "outputs/query_plan.jsonl"
qrels_jsonl = "outputs/qrels.jsonl"
summary_output_dir = "outputs/summary"
hf_dataset_dir = "outputs/hf_dataset"

# Vision adapter configuration
[vision_config]
adapter = "openai"
model = "gpt-4o"

# Judge adapter configuration
[judge_config]
adapter = "openai"
model = "gpt-4o"

# Similarity adapter configuration
[similarity_config]
adapter = "local_clip"
model_name = "openai/clip-vit-base-patch32"
```

### 2. Run the Complete Pipeline

```bash
benchmaker all --config config.toml #or use IMSEARCH_BENCHMAKER_CONFIG_PATH env variable with the path to the config file
```

This will run the entire pipeline:
1. **Preprocess**: Build `images.jsonl` from your image directory
2. **Vision**: Annotate images with tags, taxonomies, and metadata
3. **Query Plan**: Generate diverse queries with candidate images
4. **Judge**: Evaluate relevance of candidates for each query
5. **Postprocess**: Calculate similarity scores and generate summaries
6. **Upload**: Upload dataset to Hugging Face Hub

## Configuration

Configuration is done via TOML files (JSON is also supported). The framework uses a `BenchmarkConfig` class that supports:

- **Benchmark metadata**: Name, description, author information
- **Column mappings**: Customize column names for your data structure
- **File paths**: Input and output file locations
- **Adapter settings**: Configure vision, judge, and similarity adapters
- **Query planning**: Control query generation parameters
- **Hugging Face**: Repository settings for dataset upload

See `example/config.toml` for a complete configuration example.

### Sensitive Fields

Fields starting with `_` (e.g., `_hf_token`, `_openai_api_key`) are considered sensitive fields.

### Rights Map (Metadata Configuration)

The `rights_map.json` file (configured via `meta_json` in your config) allows you to assign license and DOI metadata to images during preprocessing. This is useful when images come from multiple sources with different licensing requirements.

#### Syntax

The `rights_map.json` file has the following structure:

```json
{
  "default": {
    "license": "UNKNOWN",
    "doi": "UNKNOWN"
  },
  "files": {
    "path/to/specific/image.jpg": {
      "license": "CC BY 4.0",
      "doi": "10.1234/example"
    }
  },
  "prefixes": [
    {
      "prefix": "sage/",
      "license": "UNKNOWN",
      "doi": "10.1109/ICSENS.2016.7808975"
    },
    {
      "prefix": "wildfire/",
      "license": "CC BY 4.0",
      "doi": "10.3390/f14091697"
    }
  ]
}
```

#### Matching Rules

Metadata is assigned to images using the following priority order (most specific first):

1. **Exact file match**: If an image ID appears in the `files` object, use that metadata
2. **Longest prefix match**: If the image ID starts with any prefix in the `prefixes` array, use the metadata from the longest matching prefix
3. **Default**: Use the metadata from the `default` object

#### Example

For an image with ID `sage/imagesampler-bottom-2726/image.jpg`:
- If `files` contains an exact match, use that
- Otherwise, if it starts with `sage/`, use the `sage/` prefix metadata
- Otherwise, use the `default` metadata

#### Configuration

Set the path to your rights map file in `config.toml`:

```toml
meta_json = "path/to/rights_map.json"
```

Or pass it via command line:

```bash
benchmaker preprocess --meta-json path/to/rights_map.json
```

If no `meta_json` is provided, you'll be prompted for default license and DOI values during preprocessing.

See `example/rights_map.json` for a complete example.

## CLI Commands

### Main Pipeline Commands

```bash
# Set the path to the config file so you don't have to pass it to each command
export IMSEARCH_BENCHMAKER_CONFIG_PATH="path/to/config.toml"

# Run complete pipeline
benchmaker all

# Individual steps
benchmaker preprocess
benchmaker vision
benchmaker plan
benchmaker judge
benchmaker postprocess similarity
benchmaker postprocess summary
benchmaker upload
```

### Utility Commands

```bash
# Check if image URLs are reachable
benchmaker check-urls --images-jsonl outputs/images.jsonl

# Clean intermediate files
benchmaker clean --config config.toml

# List OpenAI batches
benchmaker list-batches --config config.toml
```

### Granular Control (Vision)

For more control over the vision annotation process:

```bash
# Set the path to the config file so you don't have to pass it to each command
export IMSEARCH_BENCHMAKER_CONFIG_PATH="path/to/config.toml"

# Create batch input
benchmaker vision-make

# Submit batch
benchmaker vision-submit

# Wait for completion
benchmaker vision-wait

# Download results
benchmaker vision-download

# Parse results
benchmaker vision-parse

# Retry failed requests
benchmaker vision-retry
```

### Granular Control (Judge)

Similar granular commands are available for the judge step:

```bash
export IMSEARCH_BENCHMAKER_CONFIG_PATH="path/to/config.toml"
benchmaker judge-make
benchmaker judge-submit
benchmaker judge-wait
benchmaker judge-download
benchmaker judge-parse
benchmaker judge-retry
```

## Adapters

The framework uses an adapter pattern for extensibility. Adapters are automatically discovered and registered. **You can use different adapters for different tasks simultaneously** - for example, OpenAI for vision annotation and Google Gemini for relevance judging. Simply configure each adapter in your `config.toml` file.

### Vision Adapters

- **OpenAI**: Uses OpenAI API for image annotation with structured outputs
  - Tags, taxonomies, boolean fields
  - Confidence scores
  - Controlled vocabularies

### Judge Adapters

- **OpenAI**: Uses OpenAI API to evaluate query-image relevance
  - Binary and graded relevance labels
  - Confidence scores

### Similarity Adapters

- **Local CLIP**: Local CLIP models for similarity scoring
  - Supports any CLIP model from Hugging Face
  - No API costs

### Creating Custom Adapters

The framework supports creating custom adapters for vision annotation, relevance judging, and similarity scoring. Adapters are automatically discovered when placed in the `imsearch_benchmaker/adapters/` directory. For detailed instructions, code examples, and best practices, see [Creating Custom Adapters](imsearch_benchmaker/adapters/README.md#creating-custom-adapters).

### Mixing Adapters

You can use different adapters for different tasks in the same pipeline. For example:

```toml
# Use OpenAI for vision annotation
[vision_config]
adapter = "openai"
model = "gpt-4o"

# Use Google Gemini for relevance judging
[judge_config]
adapter = "gemini"
model = "gemini-pro"

# Use local CLIP for similarity scoring
[similarity_config]
adapter = "local_clip"
model_name = "openai/clip-vit-base-patch32"
```

Each adapter is configured independently, allowing you to choose the best service for each task based on cost, performance, or feature requirements.

## Pipeline Overview

```
┌─────────────┐
│   Images    │
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌─────────────┐
│ Preprocess  │────▶│ images.jsonl│
└─────────────┘     └──────┬───────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   Vision    │────▶ annotations.jsonl
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │ Query Plan  │────▶ query_plan.jsonl
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │    Judge    │────▶ qrels.jsonl
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │ Postprocess │────▶ qrels_with_score.jsonl
                    └──────┬──────┘      + summary/
                           │
                           ▼
                    ┌─────────────┐
                    │   Upload    │────▶ Hugging Face Hub
                    └─────────────┘
```

## Output Files

### Intermediate Files

- `images.jsonl`: Image metadata and URLs
- `seeds.jsonl`: Seed images for query generation
- `annotations.jsonl`: Vision annotations (tags, taxonomies, etc.)
- `query_plan.jsonl`: Generated queries with candidate images
- `qrels.jsonl`: Relevance judgments (query-image pairs)

### Final Outputs

- `qrels_with_score_jsonl`: QRELs with similarity scores
- `summary/`: Directory containing:
  - Dataset statistics (CSV)
  - Visualizations (PNG)
  - Cost summaries
  - Word clouds
- `hf_dataset/`: Hugging Face dataset ready for upload

## Examples

See the `example/` directory for:
- Complete configuration file (`config.toml`)
- Sample input files
- Example outputs
- Dataset card template

## Requirements

- Python >= 3.11
- Core dependencies (automatically installed):
  - see `imsearch_benchmaker/requirements.txt`
- Optional adapter dependencies:
  - see `imsearch_benchmaker/adapters/{adapter_name}/requirements.txt`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT License](LICENSE)

## Author

- **Author**: Francisco Lozano
- **Email**: francisco.lozano@northwestern.edu
- **Affiliation**: Northwestern University
- GitHub: [FranciscoLozCoding](https://github.com/FranciscoLozCoding)

## Support

For issues, questions, or contributions, please open an issue on [GitHub](https://github.com/waggle-sensor/imsearch_benchmaker).

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{imsearch_benchmaker,
  title = {Image Search Benchmark Maker},
  author = {Lozano, Francisco},
  organization = {Northwestern University},
  orcid = {0009-0003-8823-4046},
  year = {2026},
  url = {https://github.com/waggle-sensor/imsearch_benchmaker}
}
```
