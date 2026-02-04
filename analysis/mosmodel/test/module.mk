MODULE_NAME := analysis/mosmodel/test
SUBMODULES :=

# how many layouts to sample for testing:
TEST_LAYOUTS_N ?= 200
TEST_LAYOUTS_SEED ?= 1
TEST_LAYOUTS_MODE ?= moselect_plus_uniform

MODEL_EXPERIMENTS := $(TEST_EXPERIMENTS)

include $(MOSMODEL_TEMPLATE_MAKEFILE)

# ---- override MEAN_CSV_FILE recipe: build mean.csv as a random subset of the pool ----
# template.mk already defines:
#   MEAN_CSV_FILE := $(MODULE_NAME)/mean.csv
#   MODEL_MEAN_CSV_FILES := results/<exp>/mean.csv ...
# so we reuse those.

$(MEAN_CSV_FILE): $(addprefix results/,$(addsuffix /mean.csv,$(MODEL_EXPERIMENTS))) 
	python3 $(MOSMODEL_ROOT)/select_test_layouts.py \
		--inputs $^  \
		--output $@ \
		--seed $(TEST_LAYOUTS_SEED) \
		--mode $(TEST_LAYOUTS_MODE) \
		--all_layouts