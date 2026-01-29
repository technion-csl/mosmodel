# Common evaluation logic for any strategy module
# Expects:
#   MODULE_NAME := analysis/mosmodel/<strategy>
# And submodules:
#   $(MODULE_NAME)/train -> produces $(MODULE_NAME)/train/mean.csv
#   $(MODULE_NAME)/test  -> produces $(MODULE_NAME)/test/mean.csv

MOSMODEL_ROOT := analysis/mosmodel

VALIDATE_MODELS := $(MOSMODEL_ROOT)/validateModels.py
PLOT_MAX_ERRORS := $(MOSMODEL_ROOT)/plotMaxErrors.py

# Use the same polynomial config as the original mosmodel module
POLY_FILE := $(MOSMODEL_ROOT)/poly3.csv
MAX_ERRORS_PLOT_TITLE := "Max Errors"

TRAIN_MEAN := $(MODULE_NAME)/train/mean.csv
TEST_MEAN  := $(MOSMODEL_ROOT)/test/mean.csv

TEST_ERRORS_FILE := $(MODULE_NAME)/test_errors.csv
MAX_ERRORS_PLOTS := \
	$(MODULE_NAME)/linear_models_max_errors.pdf \
	$(MODULE_NAME)/polynomial_models_max_errors.pdf

$(MODULE_NAME): $(TEST_ERRORS_FILE) $(MAX_ERRORS_PLOTS)

$(TEST_ERRORS_FILE): private TRAIN_MEAN := $(MODULE_NAME)/train/mean.csv
$(TEST_ERRORS_FILE): $(TRAIN_MEAN) $(TEST_MEAN) $(LINEAR_MODELS_COEFFS)
	mkdir -p $(dir $@)
	$(VALIDATE_MODELS) \
		--train_dataset=$(TRAIN_MEAN) \
		--test_dataset=$(TEST_MEAN) \
		--output=$@ \
		--coeffs_file=$(LINEAR_MODELS_COEFFS) \
		--poly=$(POLY_FILE)

$(MAX_ERRORS_PLOTS): private TEST_ERRORS_FILE := $(MODULE_NAME)/test_errors.csv
$(MAX_ERRORS_PLOTS): $(TEST_ERRORS_FILE)
	$(PLOT_MAX_ERRORS) --errors=$(TEST_ERRORS_FILE) --plot_title=$(MAX_ERRORS_PLOT_TITLE) --output=$(@D)

$(MODULE_NAME)/clean:
	rm -f $(MODULE_NAME)/*.pdf $(MODULE_NAME)/*.csv

include $(ROOT_DIR)/common.mk

