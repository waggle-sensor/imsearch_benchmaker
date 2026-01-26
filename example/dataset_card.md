---
dataset_info:
  features:
  - name: query_id
    dtype: string
  - name: query_text
    dtype: string
  - name: image_id
    dtype: string
  - name: relevance_label
    dtype: int64
  - name: license
    dtype: string
  - name: doi
    dtype: string
  - name: tags
    sequence: string
  - name: confidence
    struct:
    - name: confounder_type
      dtype: float64
    - name: environment_type
      dtype: float64
    - name: plume_stage
      dtype: float64
    - name: viewpoint
      dtype: float64
    - name: flame_visible
      dtype: float64
    - name: lighting
      dtype: float64
  - name: environment_type
    dtype: string
  - name: confounder_type
    dtype: string
  - name: lighting
    dtype: string
  - name: flame_visible
    dtype: bool
  - name: plume_stage
    dtype: string
  - name: viewpoint
    dtype: string
  - name: summary
    dtype: string
  - name: image
    dtype: image
  splits:
  - name: train
    num_bytes: 3557246947.424
    num_examples: 4082
  download_size: 6101033867
  dataset_size: 3557246947.424
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
license: cc-by-4.0
task_categories:
- image-classification
language:
- en
tags:
- wildfire
- fire-science
- image-retrieval
- benchmark
- computer-vision
- remote-sensing
- environmental-science
size_categories:
- 1K<n<10K
---

# FireBench: A Benchmark Dataset for Fire Science Image Retrieval
COPY OF FIREBENCH, this time using the imsearch_benchmaker framework.
## Dataset Description

FireBench is a benchmark dataset for evaluating image retrieval systems in the domain of wildfire and fire science. The dataset consists of natural language queries paired with images, along with binary relevance labels indicating whether each image is relevant to the query. The dataset is designed to test retrieval systems' ability to find relevant wildfire-related images based on a query.

![Image Sample](summary/random_image_sample.png)

### Dataset Summary

FireBench contains:
- **Queries**: Natural language queries written by AI fire scientists describing wildfire phenomena
- **Images**: Real-world wildfire imagery from multiple sources (HPWREN FIgLib, The Wildfire Dataset, Sage Continuum)
- **Relevance Labels**: Binary labels (0 = not relevant, 1 = relevant) for each query-image pair assigned by AI fire scientists
- **Rich Metadata**: Comprehensive annotations including environmental conditions, smoke characteristics, lighting, viewpoint, and more
- **CLIPScore**: Pre-computed CLIP similarity scores for each query-image pair using apple/DFN5B-CLIP-ViT-H-14-378 model.

The dataset is designed to evaluate:
- Text-to-image retrieval systems
- Image search engines for fire science applications
- Multimodal understanding models
- False positive detection (distinguishing smoke from confounders like fog, haze, clouds)

Image Proportions by categories:
![Image Proportions](summary/image_proportion_donuts.png)

### Supported Tasks and Leaderboards

- **Image Retrieval**: Given a text query, retrieve relevant images from a candidate pool
- **Relevance Classification**: Classify whether an image is relevant to a given query
- **Multimodal Similarity**: Evaluate semantic similarity between text queries and images

### Languages

The dataset contains English text queries and image annotations.

## Dataset Structure

### Data Instances

Each instance in the dataset contains:

```python
{
    "query_id": "firebench_q001",
    "query_text": "Fixed long-range daytime webcam images of mountainous shrubland with no visible smoke or flames",
    "image_id": "figlib/HPWREN-FIgLib/20160604_FIRE_rm-n-mobo-c/1465063980_-01620.jpg",
    "relevance_label": 1,
    "image": <PIL.Image.Image>,  # The actual image
    "license": "CC BY 4.0",
    "doi": "2112.08598",
    "tags": ["no_smoke_visible", "no_flame_visible", "daylight", "clear_air", ...],
    "confidence": {
        "viewpoint": 0.9,
        "plume_stage": 0.95,
        "confounder_type": 0.9,
        "environment_type": 0.9
    },
    "environment_type": "mountainous",
    "confounder_type": "none",
    "lighting": "day",
    "flame_visible": false,
    "plume_stage": "none",
    "viewpoint": "fixed_long_range",
    "summary": "Clear daytime view of mountainous shrubland with dirt roads and distant ridgelines; no visible smoke or flames.",
    "clip_score": 5.337447166442871
}
```

### Data Fields

- **query_id** (string): Unique identifier for the query (e.g., "firebench_q001")
- **query_text** (string): Natural language query describing the target phenomenon
- **image_id** (string): Unique identifier for the image (relative path from source)
- **relevance_label** (int): Binary relevance label (0 = not relevant, 1 = relevant)
- **image** (Image): The actual image file
- **license** (string): License information for the image (e.g., "CC BY 4.0")
- **doi** (string): Digital Object Identifier for the source dataset
- **tags** (list of strings): Controlled vocabulary tags describing the image (12-18 tags per image)
- **confidence** (dict): Confidence scores (0-1) for categorical annotations:
  - `viewpoint`: Confidence in viewpoint classification
  - `plume_stage`: Confidence in plume stage classification
  - `confounder_type`: Confidence in confounder type classification
  - `environment_type`: Confidence in environment type classification
- **environment_type** (string): Type of environment (e.g., "mountainous", "forest", "shrubland")
- **confounder_type** (string): Type of confounder that might cause false positives (e.g., "cloud", "fog_marine_layer", "none")
- **lighting** (string): Lighting conditions (e.g., "day", "dusk", "night", "ir_nir")
- **flame_visible** (bool): Whether flames are visible in the image
- **plume_stage** (string): Stage of smoke plume (e.g., "incipient", "developing", "mature", "none")
- **viewpoint** (string): Camera viewpoint (e.g., "fixed_long_range", "handheld", "aerial")
- **summary** (string): Brief factual summary of the image (≤30 words)
- **clip_score** (float): Pre-computed CLIP similarity score between query and image

### Data Splits

The dataset is provided as a single split. Users can create their own train/validation/test splits as needed.

## Dataset Creation

### Curation Rationale

FireBench was created to address the need for a standardized benchmark for evaluating image retrieval systems in fire science applications. The dataset focuses on:

1. **Real-world diversity**: Images from multiple sources covering various environmental conditions, viewpoints, and fire stages
2. **False positive challenges**: Includes confounders (fog, haze, clouds) that are common sources of false positives in fire detection systems
3. **Expert-generated queries**: Queries written by AI fire scientists to reflect realistic search scenarios
4. **Comprehensive annotations**: Rich metadata to support both retrieval evaluation and analysis

### Source Data

The dataset combines images from three sources:

1. **HPWREN FIgLib** ([HPWREN](https://www.hpwren.ucsd.edu/FIgLib/index.html))
   - High-Performance Wireless Research and Education Network
   - Fixed long-range webcam imagery
   - DOI: 10.48550/arXiv.2112.08598

2. **The Wildfire Dataset** ([Kaggle](https://www.kaggle.com/datasets/elmadafri/the-wildfire-dataset))
   - Diverse wildfire imagery
   - DOI: 10.3390/f14091697

3. **Sage Continuum** ([Sage](https://sagecontinuum.org))
   - Cyberinfrastructure to support advanced AI research including Fire Science.
   - DOI: 10.1109/ICSENS.2016.7808975

### Annotations

#### Annotation process

1. **Vision Annotation**: Images were annotated using OpenAI's vision API (GPT-5-mini) to extract:
   - Image summaries
   - Categorical facets (viewpoint, plume_stage, lighting, confounder_type, environment_type)
   - Controlled vocabulary tags (12-18 tags per image)
   - Confidence scores for categorical annotations

2. **Query Generation**: Fire scientist queries were generated using OpenAI's text API (GPT-5-mini) based on seed images

3. **Relevance Labeling**: Binary relevance labels were assigned by AI fire scientists using OpenAI's text API (GPT-5-mini) based on query-image pairs

4. **CLIPScore Calculation**: CLIP similarity scores were computed using a local CLIP model (apple/DFN5B-CLIP-ViT-H-14-378)

#### Who are the annotators?

- **Vision annotations**: Automated using OpenAI's vision API
- **Queries**: Generated by OpenAI's text API, designed to reflect fire scientist perspectives
- **Relevance labels**: Assigned by OpenAI's text API, designed to reflect fire scientist perspectives

### Personal and Sensitive Information

The dataset contains only publicly available wildfire imagery. No personal information is included.

## Considerations for Using the Data

### Social Impact of Dataset

This dataset supports research and development of:
- Early wildfire detection systems
- Fire science research tools
- Environmental monitoring applications

### Discussion of Biases

Potential biases in the dataset:
- **Geographic bias**: Images may be biased toward regions where source datasets were collected
- **Temporal bias**: Images may reflect specific time periods when data was collected
- **Viewpoint bias**: Fixed camera viewpoints may be overrepresented
- **Environmental bias**: Certain environments (e.g., mountainous, shrubland) may be overrepresented

### Other Known Limitations

1. **Query diversity**: Queries are generated by AI models and may not fully capture all real-world search scenarios
2. **Relevance judgments**: Binary relevance labels may not capture nuanced relevance levels
3. **Image quality**: Images vary in resolution and quality depending on source
4. **Temporal coverage**: Dataset may not cover all seasons or fire conditions

## Additional Information

### Dataset Curators

The dataset was created using imsearch_benchmaker, a pipeline for creating image retrieval benchmarks.

### Licensing Information

Images in the dataset are licensed according to their source:
- **HPWREN FIgLib**: See source for licensing information
- **The Wildfire Dataset**: CC BY 4.0
- **Sage Continuum**: See source for licensing information

The dataset card and annotations are provided under CC BY 4.0.

### Citation Information

If you use this dataset, please cite:

```bibtex
@misc{sage_continuum_2026,
	author       = { Sage Continuum and Francisco Lozano },
	title        = { FireBench (Revision 03a9675) },
	year         = 2026,
	url          = { https://huggingface.co/datasets/sagecontinuum/FireBench },
	doi          = { 10.57967/hf/7454 },
	publisher    = { Hugging Face }
}
```

### Acknowledgments

We thank the creators and maintainers of:
- HPWREN FIgLib
- The Wildfire Dataset
- Sage Continuum

## Dataset Statistics

Please refer to the [EDA](summary/firebench_eda_analysis.ipynb) in the [summary/](summary/) directory.

## Hyperparameters in creating the dataset

Please refer to the [config_values.csv](summary/config_values.csv) file in the [summary/](summary/) directory for the values of the hyperparameters used in the dataset creation.

| value | description |
|-------|-------------|
FIREBENCH_NUM_SEEDS | the number of seed images to use for query generation
OPENAI_VISION_MODEL | model to use in the vision annotation
OPENAI_TEXT_MODEL | text model to use in the query generation and relevance labeling
OPENAI_BATCH_COMPLETION_WINDOW | completion window for the batch
FIREBENCH_IMAGE_DETAIL | image detail level (low, medium, high)
FIREBENCH_MAX_CANDIDATES | maximum number of candidates to generate for each query
VISION_ANNOTATION_MAX_OUTPUT_TOKENS | the maximum number of tokens for the vision annotation
VISION_ANNOTATION_REASONING_EFFORT | the reasoning effort for the vision annotation
JUDGE_MAX_OUTPUT_TOKENS | maximum number of tokens for the judge
JUDGE_REASONING_EFFORT | the reasoning effort for the judge
FIREBENCH_MAX_IMAGES_PER_BATCH | the maximum number of images per vision batch shard
FIREBENCH_MAX_QUERIES_PER_BATCH | the maximum number of queries per judge batch shard
FIREBENCH_MAX_CONCURRENT_BATCHES | the maximum number of batches to keep in flight
FIREBENCH_NEGATIVES_PER_QUERY | the total number of negatives to generate for each query
FIREBENCH_HARD_NEG | the number of hard negatives to generate for each query
FIREBENCH_NEARMISS_NEG | the number of nearmiss negatives to generate for each query
FIREBENCH_EASY_NEG | the number of easy negatives to generate for each query
FIREBENCH_RANDOM_SEED | the random seed used for reproducibility
CONTROLLED_TAG_VOCAB | the controlled tag vocabulary for the FireBench benchmark
VISON_ANNOTATION_SYSTEM_PROMPT | the system prompt for the vision annotation
VISION_ANNOTATION_USER_PROMPT | the user prompt for the vision annotation
JUDGE_SYSTEM_PROMPT | the system prompt for the judge
JUDGE_USER_PROMPT | the user prompt for the judge
VIEWPOINT | Locked taxonomy for the viewpoint of the image
PLUME_STAGE | Locked taxonomy forthe plume stage of the image
LIGHTING | Locked taxonomy forthe lighting of the image
CONFOUNDER | Locked taxonomy for the confounder of the image
ENVIRONMENT | Locked taxonomy for the environment of the image

## References

```
El-Madafri I, Peña M, Olmedo-Torre N. The Wildfire Dataset: Enhancing Deep Learning-Based Forest Fire Detection with a Diverse Evolving Open-Source Dataset Focused on Data Representativeness and a Novel Multi-Task Learning Approach. Forests. 2023; 14(9):1697. https://doi.org/10.3390/f14091697

Anshuman Dewangan, Yash Pande, Hans-Werner Braun, Frank Vernon, Ismael Perez, Ilkay Altintas, Garrison W Cottrell, and Mai H Nguyen. Figlib & smokeynet: Dataset and deep learning model for real-time wildland fire smoke detection. Remote Sensing, 14(4):1007, 2022. https://doi.org/10.48550/arXiv.2112.08598

Catlett, C. E., P. H. Beckman, R. Sankaran, and K. K. Galvin, 2017: Array of Things: A Scientific Research Instrument in the Public Way: Platform Design and Early Lessons Learned. Proceedings of the 2nd International Workshop on Science of Smart City Operations and Platforms Engineering, 26–33. https://doi.org/10.1109/ICSENS.2016.7808975
```