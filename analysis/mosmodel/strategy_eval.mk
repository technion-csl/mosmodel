MOSMODEL_ROOT := analysis/mosmodel

VALIDATE_MODELS := $(MOSMODEL_ROOT)/validateModels.py
PLOT_MAX_ERRORS := $(MOSMODEL_ROOT)/plotMaxErrors.py
COLLECT_POLYNOMIAL_COEFFICIENTS := $(MOSMODEL_ROOT)/collectPolynomialCoefficients.py
CROSS_VALIDATE := $(MOSMODEL_ROOT)/crossValidateModel.py
CALCULATE_R_SQUARES := $(MOSMODEL_ROOT)/calculateRSquares.py

#************* consts *************
MAX_ERRORS_PLOT_TITLE := "Max Errors"
TRAIN_ERRORS_FILE := $(MODULE_NAME)/train_errors.csv
TEST_ERRORS_FILE := $(MODULE_NAME)/test_errors.csv
CROSS_VALIDATION_FILE := $(MODULE_NAME)/cross_validation.csv
POLY_FILE := $(MODULE_NAME)/poly3.csv
UNIFIED_MEAN_FILE := $(MODULE_NAME)/mean.csv

POLY_COEFFICIENTS := $(MODULE_NAME)/poly_coefficients.csv
MAX_ERRORS_PLOTS := \
	$(MODULE_NAME)/linear_models_max_errors.pdf \
	$(MODULE_NAME)/polynomial_models_max_errors.pdf

TRAIN_MEAN := $(MODULE_NAME)/train/mean.csv
TEST_MEAN  := $(MOSMODEL_ROOT)/test/mean.csv

$(MODULE_NAME): $(MAX_ERRORS_PLOTS) $(TRAIN_ERRORS_FILE) $(TEST_ERRORS_FILE) $(CROSS_VALIDATION_FILE)

$(MAX_ERRORS_PLOTS): private TEST_ERRORS_FILE := $(MODULE_NAME)/test_errors.csv
$(MAX_ERRORS_PLOTS): $(TEST_ERRORS_FILE)
	$(PLOT_MAX_ERRORS) --errors=$(TEST_ERRORS_FILE) --plot_title=$(MAX_ERRORS_PLOT_TITLE) --output=$(@D)
	
$(POLY_COEFFICIENTS):
	$(COLLECT_POLYNOMIAL_COEFFICIENTS) --output=$@

$(TRAIN_ERRORS_FILE): private TRAIN_MEAN := $(MODULE_NAME)/train/mean.csv
$(TRAIN_ERRORS_FILE): $(TRAIN_MEAN) $(LINEAR_MODELS_COEFFS)
	mkdir -p $(dir $@)
	$(VALIDATE_MODELS) --train_dataset=$(TRAIN_MEAN) --test_dataset=$(TRAIN_MEAN) --output=$@ \
		--coeffs_file=$(LINEAR_MODELS_COEFFS) --poly=/dev/null

$(TEST_ERRORS_FILE): private TRAIN_MEAN := $(MODULE_NAME)/train/mean.csv
$(TEST_ERRORS_FILE): private POLY_FILE := $(MODULE_NAME)/poly3.csv
$(TEST_ERRORS_FILE): $(TRAIN_MEAN) $(TEST_MEAN) $(LINEAR_MODELS_COEFFS)
	mkdir -p $(dir $@)
	$(VALIDATE_MODELS) \
		--train_dataset=$(TRAIN_MEAN) \
		--test_dataset=$(TEST_MEAN) \
		--output=$@ \
		--coeffs_file=$(LINEAR_MODELS_COEFFS) \
		--poly=$(POLY_FILE)

$(CROSS_VALIDATION_FILE):$(UNIFIED_MEAN_FILE) 
	$(CROSS_VALIDATE) --input=$< --output=$@

$(UNIFIED_MEAN_FILE): private TRAIN_MEAN := $(MODULE_NAME)/train/mean.csv
$(UNIFIED_MEAN_FILE): $(TRAIN_MEAN) $(TEST_MEAN)
	mkdir -p $(dir $@)
	head -n 1 -q $(TRAIN_MEAN) > $@
	tail -n +2 -q $(TRAIN_MEAN) >> $@
	tail -n +2 -q $(TEST_MEAN) >> $@
	
$(MODULE_NAME)/clean:
	rm -f $(MODULE_NAME)/*.pdf $(MODULE_NAME)/*.csv

include $(ROOT_DIR)/common.mk

