MODULE_NAME := analysis/mosmodel/test
SUBMODULES :=

# how many layouts to sample for testing:
TEST_LAYOUTS_N ?= 20
TEST_LAYOUTS_SEED ?= 1
TEST_LAYOUTS_MODE ?= moselect_plus_uniform

MODEL_EXPERIMENTS := $(TEST_EXPERIMENTS)

include $(MOSMODEL_TEMPLATE_MAKEFILE)

# ---- override MEAN_CSV_FILE recipe: build mean.csv as a random subset of the pool ----
# template.mk already defines:
#   MEAN_CSV_FILE := $(MODULE_NAME)/mean.csv
#   MODEL_MEAN_CSV_FILES := results/<exp>/mean.csv ...
# so we reuse those.

$(MEAN_CSV_FILE): $(MODEL_MEAN_CSV_FILES) $(MOSMODEL_ROOT)/select_test_layouts.py
	python3 $(MOSMODEL_ROOT)/select_test_layouts.py \
		--inputs $(MODEL_MEAN_CSV_FILES) \
		--output $@ \
		--num_total $(TEST_LAYOUTS_N) \
		--seed $(TEST_LAYOUTS_SEED) \
		--mode $(TEST_LAYOUTS_MODE)