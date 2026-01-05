SHELL := /bin/bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c

# ---- Preprocess inputs ----
IMAGE_ROOT_DIR ?= /tmp/FireBench/images
META_JSON      ?= /tmp/FireBench/images/rights_map.json

# ---- Inputs you provide ----
IMAGES_JSONL ?= /Volumes/data/inputs/images.jsonl
SEEDS_JSONL  ?= /Volumes/data/inputs/seeds.jsonl

# ---- Intermediate / outputs ----
VISION_BATCH_JSONL ?= /Volumes/data/outputs/vision_batch.jsonl
VISION_OUT_JSONL   ?= /Volumes/data/outputs/vision_out.jsonl
VISION_ERR_JSONL   ?= /Volumes/data/outputs/vision_err.jsonl
VISION_RETRY_BATCH_JSONL ?= /Volumes/data/outputs/vision_retry/batch.jsonl
ANNOTATIONS_JSONL  ?= /Volumes/data/outputs/annotations.jsonl

QUERY_PLAN_JSONL   ?= /Volumes/data/outputs/query_plan.jsonl

JUDGE_BATCH_JSONL  ?= /Volumes/data/outputs/judge_batch.jsonl
JUDGE_OUT_JSONL    ?= /Volumes/data/outputs/judge_out.jsonl
JUDGE_ERR_JSONL    ?= /Volumes/data/outputs/judge_err.jsonl
JUDGE_RETRY_BATCH_JSONL ?= /Volumes/data/outputs/judge_retry/batch.jsonl
QRELS_JSONL        ?= /Volumes/data/outputs/firebench_qrels.jsonl
QRELS_WITH_CLIPSCORE_JSONL ?= /Volumes/data/outputs/firebench_qrels_with_clipscore.jsonl
SUMMARY_OUTPUT_DIR  ?= /Volumes/data/outputs/summary

# ---- Batch ID tracking ----
VISION_BATCH_ID_FILE ?= /Volumes/data/.vision_batch_id
VISION_RETRY_BATCH_ID_FILE ?= /Volumes/data/.vision_retry_batch_id
JUDGE_BATCH_ID_FILE  ?= /Volumes/data/.judge_batch_id
JUDGE_RETRY_BATCH_ID_FILE ?= /Volumes/data/.judge_retry_batch_id

# ---- Hugging Face ----
HF_DATASET_DIR ?= /Volumes/data/outputs/hf_dataset

# ---- Tools ----
PY ?= python

.PHONY: help env-check \
        preprocess vision-make vision-submit vision-wait vision-download vision-parse vision \
        vision-retry-make vision-retry-submit vision-retry-wait vision-retry-download vision-retry-parse vision-retry \
        plan \
        judge-make judge-submit judge-wait judge-download judge-parse judge \
        judge-retry-make judge-retry-submit judge-retry-wait judge-retry-download judge-retry-parse judge-retry \
        summary clipscore huggingface\
        all clean clean-outputs list-active list-all

help:
	@echo "Targets:"
	@echo "  env-check        Validate OPENAI_API_KEY is set"
	@echo ""
	@echo "  preprocess       Preprocess a folder of images to create images.jsonl and seeds.jsonl"
	@echo ""
	@echo "  vision-make      Create vision_batch.jsonl from images.jsonl"
	@echo "  vision-submit    Submit vision batch; writes batch id to .vision_batch_id"
	@echo "  vision-wait      Wait for vision batch completion"
	@echo "  vision-download  Download vision output/error JSONL"
	@echo "  vision-parse     Parse vision output -> annotations.jsonl"
	@echo "  vision           Runs: vision-make -> vision-submit -> vision-wait -> vision-download -> vision-parse"
	@echo ""
	@echo "  vision-retry-make      Create retry batch from vision_err.jsonl (failed requests only)"
	@echo "  vision-retry-submit    Submit vision retry batch; writes batch id to .vision_retry_batch_id"
	@echo "  vision-retry-wait      Wait for vision retry batch completion"
	@echo "  vision-retry-download  Download vision retry output/error JSONL"
	@echo "  vision-retry-parse     Parse vision retry output -> annotations.jsonl (appends to existing)"
	@echo "  vision-retry           Runs: vision-retry-make -> vision-retry-submit -> vision-retry-wait -> vision-retry-download -> vision-retry-parse"
	@echo ""
	@echo "  plan             Build query_plan.jsonl from annotations.jsonl + seeds.jsonl"
	@echo ""
	@echo "  judge-make       Create judge_batch.jsonl from query_plan.jsonl + annotations.jsonl"
	@echo "  judge-submit     Submit judge batch; writes batch id to .judge_batch_id"
	@echo "  judge-wait       Wait for judge batch completion"
	@echo "  judge-download   Download judge output/error JSONL"
	@echo "  judge-parse      Parse judge output -> firebench_qrels.jsonl"
	@echo "  judge            Runs: judge-make -> judge-submit -> judge-wait -> judge-download -> judge-parse"
	@echo ""
	@echo "  judge-retry-make      Create retry batch from judge_err.jsonl (failed requests only)"
	@echo "  judge-retry-submit    Submit judge retry batch; writes batch id to .judge_retry_batch_id"
	@echo "  judge-retry-wait      Wait for judge retry batch completion"
	@echo "  judge-retry-download  Download judge retry output/error JSONL"
	@echo "  judge-retry-parse     Parse judge retry output -> firebench_qrels.jsonl (appends to existing)"
	@echo "  judge-retry           Runs: judge-retry-make -> judge-retry-submit -> judge-retry-wait -> judge-retry-download -> judge-retry-parse"
	@echo ""
	@echo "  all              Runs: preprocess -> vision -> plan -> judge -> postprocess -> huggingface"
	@echo ""
	@echo "Postprocessing:"
	@echo "  clipscore        Calculate CLIPScore for each row in qrels JSONL and add it as a column"
	@echo "  summary          Generate a Exploratory Data Analysis (EDA) of the dataset"
	@echo "  postprocess      Runs: clipscore -> summary"
	@echo ""
	@echo "huggingface      Prepare and optionally upload the dataset to Hugging Face Hub"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean            Remove all outputs that were only used to generate the final outputs"
	@echo ""
	@echo "Helper commands:"
	@echo "  list-active     List OpenAI Active batches"
	@echo "  list-all        List All OpenAI batches"

env-check:
	@if [[ -z "$${OPENAI_API_KEY:-}" ]]; then \
		echo "ERROR: OPENAI_API_KEY is not set"; \
		exit 1; \
	fi
	@echo "OPENAI_API_KEY set ✅"
	@echo "OPENAI_VISION_MODEL=$${OPENAI_VISION_MODEL:-<default in script>}"
	@echo "OPENAI_TEXT_MODEL=$${OPENAI_TEXT_MODEL:-<default in script>}"

# ---------------------------
# Preprocess phase
# ---------------------------

preprocess: env-check
	@if [[ -n "$(META_JSON)" ]]; then \
		echo "Using metadata file: $(META_JSON)"; \
		$(PY) preprocess.py \
			--input-dir $(IMAGE_ROOT_DIR) \
			--out-jsonl $(IMAGES_JSONL) \
			--out-seeds-jsonl $(SEEDS_JSONL) \
			--meta-json $(META_JSON); \
	else \
		echo "No META_JSON provided; using interactive/default license settings"; \
		$(PY) preprocess.py \
			--input-dir $(IMAGE_ROOT_DIR) \
			--out-jsonl $(IMAGES_JSONL) \
			--out-seeds-jsonl $(SEEDS_JSONL) 
	fi
	@echo "✅ Preprocess complete -> $(IMAGES_JSONL), $(SEEDS_JSONL)"

# ---------------------------
# Vision phase
# ---------------------------

vision-make: env-check $(IMAGES_JSONL)
	$(PY) firebench.py make-vision-batch \
		--images-jsonl $(IMAGES_JSONL) \
		--out-jsonl $(VISION_BATCH_JSONL)

vision-submit: env-check $(VISION_BATCH_JSONL)
	# Capture batch_id(s) printed by the tool into a file
	@$(PY) firebench.py submit \
		--input-jsonl $(VISION_BATCH_JSONL) \
		--stage vision | tee /dev/stderr | jq -r 'if .sharded then (.batch_ids | join(",")) else .batch_id end' > $(VISION_BATCH_ID_FILE)
	@echo "Vision batch id(s) saved to $(VISION_BATCH_ID_FILE): $$(cat $(VISION_BATCH_ID_FILE))"

vision-wait: env-check $(VISION_BATCH_ID_FILE)
	$(PY) firebench.py wait --batch-id "$$(cat $(VISION_BATCH_ID_FILE))"

vision-download: env-check $(VISION_BATCH_ID_FILE)
	$(PY) firebench.py download-output \
		--batch-id "$$(cat $(VISION_BATCH_ID_FILE))" \
		--out-output-jsonl $(VISION_OUT_JSONL) \
		--out-error-jsonl $(VISION_ERR_JSONL)

vision-parse: $(VISION_OUT_JSONL) $(IMAGES_JSONL)
	$(PY) firebench.py parse-vision \
		--batch-output-jsonl $(VISION_OUT_JSONL) \
		--out-annotations-jsonl $(ANNOTATIONS_JSONL) \
		--images-jsonl $(IMAGES_JSONL)

vision: vision-make vision-submit vision-wait vision-download vision-parse
	@echo "✅ Vision phase complete -> $(ANNOTATIONS_JSONL)"

# ---------------------------
# Vision retry phase (for failed requests)
# ---------------------------

vision-retry-make: env-check $(VISION_ERR_JSONL) $(IMAGES_JSONL)
	@if [[ ! -s $(VISION_ERR_JSONL) ]]; then \
		echo "ERROR: $(VISION_ERR_JSONL) is empty or does not exist"; \
		echo "Run 'vision-download' first to get error file"; \
		exit 1; \
	fi
	$(PY) firebench.py resubmit-vision-errors \
		--error-jsonl $(VISION_ERR_JSONL) \
		--images-jsonl $(IMAGES_JSONL) \
		--out-jsonl $(VISION_RETRY_BATCH_JSONL)
	@echo "✅ Created retry batch -> $(VISION_RETRY_BATCH_JSONL)"

vision-retry-submit: env-check $(VISION_RETRY_BATCH_JSONL)
	# Capture batch_id(s) printed by the tool into a file
	@$(PY) firebench.py submit \
		--input-jsonl $(VISION_RETRY_BATCH_JSONL) \
		--stage vision | tee /dev/stderr | jq -r 'if .sharded then (.batch_ids | join(",")) else .batch_id end' > $(VISION_RETRY_BATCH_ID_FILE)
	@echo "Vision retry batch id(s) saved to $(VISION_RETRY_BATCH_ID_FILE): $$(cat $(VISION_RETRY_BATCH_ID_FILE))"

vision-retry-wait: env-check $(VISION_RETRY_BATCH_ID_FILE)
	$(PY) firebench.py wait --batch-id "$$(cat $(VISION_RETRY_BATCH_ID_FILE))"

vision-retry-download: env-check $(VISION_RETRY_BATCH_ID_FILE)
	$(PY) firebench.py download-output \
		--batch-id "$$(cat $(VISION_RETRY_BATCH_ID_FILE))" \
		--out-output-jsonl $(VISION_OUT_JSONL).retry \
		--out-error-jsonl $(VISION_ERR_JSONL).retry

vision-retry-parse: $(VISION_OUT_JSONL).retry $(IMAGES_JSONL)
	$(PY) firebench.py parse-vision \
		--batch-output-jsonl $(VISION_OUT_JSONL).retry \
		--out-annotations-jsonl $(ANNOTATIONS_JSONL).retry \
		--images-jsonl $(IMAGES_JSONL)
	@echo "✅ Retry annotations written to $(ANNOTATIONS_JSONL).retry"
	@echo "   Merge with main annotations: cat $(ANNOTATIONS_JSONL).retry >> $(ANNOTATIONS_JSONL)"

vision-retry: vision-retry-make vision-retry-submit vision-retry-wait vision-retry-download vision-retry-parse
	@echo "✅ Vision retry phase complete -> $(ANNOTATIONS_JSONL).retry"

# ---------------------------
# Candidate pool plan
# ---------------------------

plan: $(ANNOTATIONS_JSONL) $(SEEDS_JSONL)
	$(PY) build_query_plan.py \
		--annotations $(ANNOTATIONS_JSONL) \
		--seeds $(SEEDS_JSONL) \
		--out $(QUERY_PLAN_JSONL)
	@echo "✅ Built query plan -> $(QUERY_PLAN_JSONL)"

# ---------------------------
# Judge phase
# ---------------------------

judge-make: env-check $(QUERY_PLAN_JSONL) $(ANNOTATIONS_JSONL)
	$(PY) firebench.py make-judge-batch \
		--query-plan-jsonl $(QUERY_PLAN_JSONL) \
		--annotations-jsonl $(ANNOTATIONS_JSONL) \
		--out-jsonl $(JUDGE_BATCH_JSONL)

judge-submit: env-check $(JUDGE_BATCH_JSONL)
	# Capture batch_id(s) printed by the tool into a file
	@$(PY) firebench.py submit \
		--input-jsonl $(JUDGE_BATCH_JSONL) \
		--stage judge | tee /dev/stderr | jq -r 'if .sharded then (.batch_ids | join(",")) else .batch_id end' > $(JUDGE_BATCH_ID_FILE)
	@echo "Judge batch id(s) saved to $(JUDGE_BATCH_ID_FILE): $$(cat $(JUDGE_BATCH_ID_FILE))"

judge-wait: env-check $(JUDGE_BATCH_ID_FILE)
	$(PY) firebench.py wait --batch-id "$$(cat $(JUDGE_BATCH_ID_FILE))"

judge-download: env-check $(JUDGE_BATCH_ID_FILE)
	$(PY) firebench.py download-output \
		--batch-id "$$(cat $(JUDGE_BATCH_ID_FILE))" \
		--out-output-jsonl $(JUDGE_OUT_JSONL) \
		--out-error-jsonl $(JUDGE_ERR_JSONL)

judge-parse: $(JUDGE_OUT_JSONL) $(ANNOTATIONS_JSONL)
	$(PY) firebench.py parse-judge \
		--batch-output-jsonl $(JUDGE_OUT_JSONL) \
		--out-qrels-jsonl $(QRELS_JSONL) \
		--annotations-jsonl $(ANNOTATIONS_JSONL)

judge: judge-make judge-submit judge-wait judge-download judge-parse
	@echo "✅ Judge phase complete -> $(QRELS_JSONL)"

# ---------------------------
# Judge retry phase (for failed requests)
# ---------------------------

judge-retry-make: env-check $(JUDGE_ERR_JSONL) $(QUERY_PLAN_JSONL) $(ANNOTATIONS_JSONL)
	@if [[ ! -s $(JUDGE_ERR_JSONL) ]]; then \
		echo "ERROR: $(JUDGE_ERR_JSONL) is empty or does not exist"; \
		echo "Run 'judge-download' first to get error file"; \
		exit 1; \
	fi
	$(PY) firebench.py resubmit-judge-errors \
		--error-jsonl $(JUDGE_ERR_JSONL) \
		--query-plan-jsonl $(QUERY_PLAN_JSONL) \
		--annotations-jsonl $(ANNOTATIONS_JSONL) \
		--out-jsonl $(JUDGE_RETRY_BATCH_JSONL)
	@echo "✅ Created retry batch -> $(JUDGE_RETRY_BATCH_JSONL)"

judge-retry-submit: env-check $(JUDGE_RETRY_BATCH_JSONL)
	# Capture batch_id(s) printed by the tool into a file
	@$(PY) firebench.py submit \
		--input-jsonl $(JUDGE_RETRY_BATCH_JSONL) \
		--stage judge | tee /dev/stderr | jq -r 'if .sharded then (.batch_ids | join(",")) else .batch_id end' > $(JUDGE_RETRY_BATCH_ID_FILE)
	@echo "Judge retry batch id(s) saved to $(JUDGE_RETRY_BATCH_ID_FILE): $$(cat $(JUDGE_RETRY_BATCH_ID_FILE))"

judge-retry-wait: env-check $(JUDGE_RETRY_BATCH_ID_FILE)
	$(PY) firebench.py wait --batch-id "$$(cat $(JUDGE_RETRY_BATCH_ID_FILE))"

judge-retry-download: env-check $(JUDGE_RETRY_BATCH_ID_FILE)
	$(PY) firebench.py download-output \
		--batch-id "$$(cat $(JUDGE_RETRY_BATCH_ID_FILE))" \
		--out-output-jsonl $(JUDGE_OUT_JSONL).retry \
		--out-error-jsonl $(JUDGE_ERR_JSONL).retry

judge-retry-parse: $(JUDGE_OUT_JSONL).retry $(ANNOTATIONS_JSONL)
	$(PY) firebench.py parse-judge \
		--batch-output-jsonl $(JUDGE_OUT_JSONL).retry \
		--out-qrels-jsonl $(QRELS_JSONL).retry \
		--annotations-jsonl $(ANNOTATIONS_JSONL)
	@echo "✅ Retry qrels written to $(QRELS_JSONL).retry"
	@echo "   Merge with main qrels: cat $(QRELS_JSONL).retry >> $(QRELS_JSONL)"

judge-retry: judge-retry-make judge-retry-submit judge-retry-wait judge-retry-download judge-retry-parse
	@echo "✅ Judge retry phase complete -> $(QRELS_JSONL).retry"

# ---------------------------
# Postprocessing
# ---------------------------

clipscore: $(QRELS_JSONL) $(IMAGES_JSONL)
	$(PY) postprocess.py calculate-clipscore \
		--qrels-jsonl $(QRELS_JSONL) \
		--output-jsonl $(QRELS_WITH_CLIPSCORE_JSONL) \
		--images-jsonl $(IMAGES_JSONL) \
		--progress-interval 10
	@echo "✅ CLIPScore calculation complete!"
	@echo "   Output: $(QRELS_WITH_CLIPSCORE_JSONL)"

summary: $(QRELS_WITH_CLIPSCORE_JSONL) $(IMAGES_JSONL)
	$(PY) postprocess.py generate-summary \
		--qrels-jsonl $(QRELS_WITH_CLIPSCORE_JSONL) \
		--output-dir $(SUMMARY_OUTPUT_DIR) \
		--images-jsonl $(IMAGES_JSONL)

postprocess: clipscore summary
	@echo "✅ Postprocessing complete!"

# ---------------------------
# upload
# ---------------------------
huggingface: $(QRELS_WITH_CLIPSCORE_JSONL)
	$(PY) postprocess.py huggingface \
		--qrels-jsonl $(QRELS_WITH_CLIPSCORE_JSONL) \
		--images-jsonl $(IMAGES_JSONL) \
		--output-dir $(HF_DATASET_DIR) \
		--progress-interval 100

# ---------------------------
# Full pipeline
# ---------------------------

all: preprocess vision plan judge postprocess huggingface
	@echo "🎉 FireBench pipeline complete!"
	@echo "Annotations: $(ANNOTATIONS_JSONL)"
	@echo "Query plan:   $(QUERY_PLAN_JSONL)"
	@echo "Qrels:       $(QRELS_JSONL)"
	@echo "Qrels with CLIPScore: $(QRELS_WITH_CLIPSCORE_JSONL)"
	@echo "Summary:     $(SUMMARY_OUTPUT_DIR)"
	@echo "Hugging Face: $(HF_DATASET_DIR)"

# ---------------------------
# Cleanup
# ---------------------------

clean:
	rm -f $(VISION_BATCH_JSONL) $(VISION_ERR_JSONL) $(VISION_RETRY_BATCH_JSONL) $(VISION_ERR_JSONL).retry \
	      $(QUERY_PLAN_JSONL) $(JUDGE_BATCH_JSONL) $(JUDGE_OUT_JSONL) $(JUDGE_ERR_JSONL) \
	      $(JUDGE_RETRY_BATCH_JSONL) $(JUDGE_OUT_JSONL).retry $(JUDGE_ERR_JSONL).retry \
		  $(VISION_BATCH_ID_FILE) $(VISION_RETRY_BATCH_ID_FILE) $(JUDGE_BATCH_ID_FILE) $(JUDGE_RETRY_BATCH_ID_FILE) \
		  $(QRELS_JSONL) $(VISION_OUT_JSONL) $(VISION_OUT_JSONL).retry
	@echo "✅ Cleanup complete!"

# ---------------------------
# Helper commands
# ---------------------------

list-active:
	$(PY) firebench.py list-batches \
		--active-only

list-all:
	$(PY) firebench.py list-batches