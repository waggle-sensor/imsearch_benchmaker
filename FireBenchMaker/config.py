"""
config.py

Configuration for the FireBench benchmark.

Environment:
  FIREBENCH_NUM_SEEDS: the number of seed images to use for query generation (default: 100)
  OPENAI_API_KEY                required
  OPENAI_VISION_MODEL           vision model to use in the vision annotation (default: gpt-4.1-mini)
  OPENAI_TEXT_MODEL             text model to use in the query generation and relevance labeling (default: gpt-4.1-mini)
  OPENAI_BATCH_COMPLETION_WINDOW completion window for the batch default: 24h
  FIREBENCH_IMAGE_DETAIL        image detail level default: low  (low saves tokens)
  FIREBENCH_MAX_CANDIDATES      maximum number of candidates to generate for each query default: 80   (text payload guardrail)
  VISION_ANNOTATION_MAX_OUTPUT_TOKENS: the maximum number of tokens for the vision annotation
  VISION_ANNOTATION_REASONING_EFFORT: the reasoning effort for the vision annotation
  JUDGE_MAX_OUTPUT_TOKENS: the maximum number of tokens for the judge
  JUDGE_REASONING_EFFORT: the reasoning effort for the judge
  FIREBENCH_MAX_IMAGES_PER_BATCH: the maximum number of images per vision batch shard
  FIREBENCH_MAX_QUERIES_PER_BATCH: the maximum number of queries per judge batch shard
  FIREBENCH_MAX_CONCURRENT_BATCHES: the maximum number of batches to keep in flight
  FIREBENCH_IMAGE_BASE_URL      required     (base URL for public image access, e.g., "https://example.com/images")
  FIREBENCH_NEGATIVES_PER_QUERY: the total number of negatives to generate for each query
  FIREBENCH_HARD_NEG: the number of hard negatives to generate for each query
  FIREBENCH_NEARMISS_NEG: the number of nearmiss negatives to generate for each query
  FIREBENCH_EASY_NEG: the number of easy negatives to generate for each query
  FIREBENCH_RANDOM_SEED: the random seed used for reproducibility
  HF_TOKEN: Hugging Face token for authentication
  HF_REPO_ID: Hugging Face repository ID (e.g., "username/dataset-name")
  HF_PRIVATE: whether to create a private repository (default: false)
  CONTROLLED_TAG_VOCAB: the controlled tag vocabulary for the FireBench benchmark
  VISON_ANNOTATION_SYSTEM_PROMPT: the system prompt for the vision annotation
  VISION_ANNOTATION_USER_PROMPT: the user prompt for the vision annotation
  JUDGE_SYSTEM_PROMPT: the system prompt for the judge
  JUDGE_USER_PROMPT: the user prompt for the judge

Locked taxonomy for the FireBench benchmark:
  - VIEWPOINT: the viewpoint of the image
  - PLUME_STAGE: the plume stage of the image
  - LIGHTING: the lighting of the image
  - CONFOUNDER: the confounder of the image
  - ENVIRONMENT: the environment of the image
"""
import os

# Preprocessing Configuration
NUM_SEEDS = int(os.getenv("FIREBENCH_NUM_SEEDS", "100")) # number of seed images to use for query generation

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "") # OpenAI API key for authentication
VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-5-mini") # vision model to use
TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-5-mini") # text model to use
COMPLETION_WINDOW = os.getenv("OPENAI_BATCH_COMPLETION_WINDOW", "24h") # completion window for the batch
IMAGE_DETAIL = os.getenv("FIREBENCH_IMAGE_DETAIL", "low")  # "low" saves tokens
MAX_CANDIDATES = int(os.getenv("FIREBENCH_MAX_CANDIDATES", "100")) # maximum number of candidates to generate for each query

# Vision model configuration
VISION_ANNOTATION_MAX_OUTPUT_TOKENS = int(os.getenv("VISION_ANNOTATION_MAX_OUTPUT_TOKENS", "4000"))
VISION_ANNOTATION_REASONING_EFFORT = os.getenv("VISION_ANNOTATION_REASONING_EFFORT", "low")

# Judge model configuration
JUDGE_MAX_OUTPUT_TOKENS = int(os.getenv("JUDGE_MAX_OUTPUT_TOKENS", "8000"))
JUDGE_REASONING_EFFORT = os.getenv("JUDGE_REASONING_EFFORT", "medium")

# Batch sharding configuration (to avoid 5M enqueued token limit)
# Conservative estimate: ~4000 tokens per image (input + output)
# Max images per batch: 5,000,000 / 4,000 ≈ 1,250, but use 800-1000 to be safe
MAX_IMAGES_PER_BATCH = int(os.getenv("FIREBENCH_MAX_IMAGES_PER_BATCH", "900"))  # max images per vision batch shard
MAX_QUERIES_PER_BATCH = int(os.getenv("FIREBENCH_MAX_QUERIES_PER_BATCH", "600"))  # max queries per judge batch shard
MAX_CONCURRENT_BATCHES = int(os.getenv("FIREBENCH_MAX_CONCURRENT_BATCHES", "1"))  # max batches to keep in flight

# Image URL Configuration
IMAGE_BASE_URL = os.getenv("FIREBENCH_IMAGE_BASE_URL", "https://web.lcrc.anl.gov/public/waggle/datasets/FireBench/images")  # Base URL for public image access (e.g., "https://example.com/images")

# Query Planning Configuration
NEG_TOTAL = int(os.getenv("FIREBENCH_NEGATIVES_PER_QUERY", "40")) # total number of negatives to generate for each query
NEG_HARD = int(os.getenv("FIREBENCH_HARD_NEG", "25")) # number of hard negatives to generate for each query
NEG_NEARMISS = int(os.getenv("FIREBENCH_NEARMISS_NEG", "10")) # number of nearmiss negatives to generate for each query
NEG_EASY = int(os.getenv("FIREBENCH_EASY_NEG", "5")) # number of easy negatives to generate for each query

# RANDOM SEED
RANDOM_SEED = int(os.getenv("FIREBENCH_RANDOM_SEED", "42"))

# Hugging Face Configuration
HF_TOKEN = os.getenv("HF_TOKEN", "")  # Hugging Face token for authentication
HF_REPO_ID = os.getenv("HF_REPO_ID", "sagecontinuum/FireBench")  # Hugging Face repository ID (e.g., "username/dataset-name")
HF_PRIVATE = os.getenv("HF_PRIVATE", "false").lower() == "true"  # Whether to create a private repository

# CLIPScore Configuration
# Use safetensors=True to avoid torch.load security requirement (torch >= 2.6)
# This allows the model to work with older torch versions (e.g., 2.2.x)
CLIP_ADAPTER = os.getenv("CLIP_ADAPTER", "local")
CLIP_ADAPTER_ARGS = os.getenv("CLIP_ADAPTER_ARGS", '{"model": "apple/DFN5B-CLIP-ViT-H-14-378", "device": "cpu", "use_safetensors": true}')

# Enumerations (locked taxonomy)
VIEWPOINT = ["fixed_long_range", "handheld", "aerial", "other", "unknown"]
PLUME_STAGE = ["incipient", "developing", "mature", "residual", "none", "unknown"]
LIGHTING = ["day", "dusk", "night", "ir_nir", "unknown"]
CONFOUNDER = ["cloud", "fog_marine_layer", "dust", "haze", "sun_glare", "none", "unknown"]
ENVIRONMENT = [
    "forest", "grassland", "shrubland", "mountainous", "urban_wui", "coastal",
    "desert", "agricultural", "water", "other", "unknown"
]

# Controlled Tag field Vocabulary
CONTROLLED_TAG_VOCAB = [
  # --- Fire / smoke presence ---
  "smoke_present","no_smoke_visible","flame_visible","no_flame_visible","glow_visible","no_glow_visible",
  "active_fire_suspected","no_fire_visible",

  # --- Smoke morphology / dynamics ---
  "thin_smoke","moderate_smoke","dense_smoke","diffuse_smoke","patchy_smoke",
  "rising_plume","horizontal_smoke","wind_sheared_plume","columnar_plume","billowing_plume",
  "low_lying_smoke","valley_smoke","ridge_smoke","canopy_level_smoke",
  "smoke_source_visible","smoke_source_not_visible",
  "multiple_plumes","single_plume",

  # --- Fire behavior cues (visual only) ---
  "flame_front","spotting_suspected","torching_suspected","crown_fire_suspected","surface_fire_suspected",
  "smoldering_suspected",

  # --- Terrain / land cover ---
  "forest","conifer_forest","deciduous_forest","mixed_forest",
  "grassland","shrubland","chaparral","woodland",
  "mountainous","hills","flat_terrain","canyon","valley","ridgeline",
  "coastal","desert","agricultural","wetland","water_body",
  "urban","suburban","wildland_urban_interface",

  # --- Scene structures / human features ---
  "road_visible","trail_visible","powerlines_visible","buildings_visible","industrial_area",
  "smokestack_visible","vehicle_visible","volcano_visible",

  # --- Atmospherics / visibility ---
  "clear_air","low_contrast","high_contrast","reduced_visibility","good_visibility",
  "backlit","frontlit","side_lit","sun_in_frame","sun_near_horizon",
  "glare_present","lens_flare_present",

  # --- Confounders (critical for false positives) ---
  "fog_present","marine_layer_present","haze_present","dust_present","clouds_present","overcast",
  "cloud_deck","cloud_shadows","blown_out_sky",
  "steam_present","industrial_plume_present",

  # --- Weather cues (visual only) ---
  "windy_suspected","calm_suspected","precipitation_visible","rain_visible","snow_visible",

  # --- Time / lighting ---
  "daylight","dusk_twilight","night","artificial_lights_visible",
  "ir_nir_imagery","thermal_like",

  # --- Camera / viewpoint cues ---
  "fixed_camera","handheld_camera","aerial_view",
  "long_range_view","telephoto_view","wide_angle_view",
  "horizon_visible","sky_dominant","ground_dominant",
  "stable_frame","motion_blur","camera_shake", "thermal_view",

  # --- Composition / scale ---
  "close_up","mid_range","far_range",
  "foreground_trees","foreground_structures","foreground_terrain",
  "background_mountains","background_ocean","background_city",

  # --- Smoke color / optical cues ---
  "white_smoke","gray_smoke","dark_smoke","brown_smoke",
  "transparent_smoke","opaque_smoke",

  # --- Event context cues (visual) ---
  "multiple_smoke_sources","isolated_smoke_source",
  "smoke_over_ridge","smoke_in_valley","smoke_near_horizon",
  "plume_dispersing","plume_intensifying_suspected",

  # --- Quality / artifacts ---
  "low_resolution","high_resolution","compression_artifacts","sensor_noise",
]

# Prompt Configuration
VISON_ANNOTATION_SYSTEM_PROMPT = (
    "You are labeling wildfire imagery for a fire-science retrieval benchmark.\n"
    "Output MUST be valid JSON matching the schema. Do not include extra keys.\n"
    "Use the allowed taxonomy values exactly.\n"
    "Be conservative: if unsure, choose 'unknown'.\n" +
    "Tagging rules:\n"
    "- Prefer tags that help retrieval and false-positive analysis (smoke vs fog/haze/cloud/glare).\n"
    "- Avoid redundant near-duplicates; pick the most specific tag.\n"
)
VISION_ANNOTATION_USER_PROMPT = (
    "Analyze the image and output JSON with:\n"
    "- summary: <= 30 words, factual, no speculation\n"
    "- viewpoint, plume_stage, flame_visible, lighting, confounder_type, environment_type\n"
    "- tags: choose 12-18 tags ONLY from the provided enum list for the tags field\n"
    "- confidence: 0..1 per field (viewpoint, plume_stage, confounder_type, environment_type)\n"
)
JUDGE_SYSTEM_PROMPT = (
    "You are a fire scientist creating and judging an image-retrieval benchmark.\n"
    "You will receive a query seed (1-3 seed images described by text) and a list of candidates.\n"
    "DO NOT use file names, dataset sources, IDs, or rights info for relevance.\n"
    "Judge relevance ONLY from the summaries + facet metadata provided.\n"
    "Output MUST be valid JSON matching the schema.\n"
    "Binary relevance only: 1 relevant, 0 not relevant.\n"
)
JUDGE_USER_PROMPT = (
    "Tasks:\n"
    "1) Write a realistic fire-scientist query_text (what someone would ask for in a search engine) describing the target phenomenon.\n"
    "2) Label each candidate image as relevant (1) or not relevant (0) to that query.\n"
    "Be consistent with the query and try to make the query as concise as possible.\n"
)
