# Image Search Benchmark Maker
Image Search Benchmark Maker is a tool for creating image search benchmarks. Creates structured image search evaluation datasets by:
- Annotating images with vision models
- Generating search queries from seed images
- Creating relevance judgments for query-image pairs
- Calculating a similarity score for query-image pairs using a CLIP model.
>NOTE: This is a work in progress and is not yet ready for use. The plan is to abstract the FireBenchMaker code to create a framework that can be used to create other benchmarks.

### Benchmarks
## FireBenchMaker
[FireBenchMaker](FireBenchMaker) is a tool for creating a benchmark for image search that benchmarks the performance of the system against retrieving "Fire Science" related images.

## CLIPScore
[CLIPScore](clipscore) is a framework for calculating the CLIPScore for a query-image pair.


# TODO
- [ ] Abstract the code to create a framework that can be used to create other benchmarks.
    - FireBench can be used as a template and if needed, create adapters for specific implementations.