# Image Search Benchmark Maker
Image Search Benchmark Maker is a tool for creating image search benchmarks. It creates structured image search evaluation datasets by:
- Annotating images with vision models
- Generating search queries from seed images
- Creating relevance judgments for query-image pairs
- Calculating a similarity score for query-image pairs using a CLIP model.
>NOTE: This is a work in progress and is not yet ready for use.

## Framework
The `imsearch_benchmaker/` package provides the shared framework and adapters for:
- Vision annotation (`imsearch_benchmaker/framework/vision.py`)
- Query generation + relevance judging (`imsearch_benchmaker/framework/judge.py`)
- CLIP scoring (`imsearch_benchmaker/framework/clip.py`, backed by `clipscore/`)

Adapters live in `imsearch_benchmaker/adapters/`. Right now the OpenAI adapter is implemented for
vision + judge (see `imsearch_benchmaker/adapters/openai/`).

### Benchmarks
## FireBenchMaker
[FireBenchMaker](FireBenchMaker) is a tool for creating a benchmark for image search that benchmarks the performance of the system against retrieving "Fire Science" related images.

## CLIPScore
[CLIPScore](clipscore) is a framework for calculating the CLIPScore for a query-image pair and is
used by `imsearch_benchmaker/framework/clip.py`.


# TODO
- [ ] Extract FireBench-specific logic into new adapters/configs as we expand benchmarks.