MODULE_NAME := analysis/linear_models_coeffs
SUBMODULES := 

BUILD_LINEAR_MODELS_COEFFS := $(ROOT_DIR)/$(MODULE_NAME)/buildLinearModelsCoeffs.py
LINEAR_MODELS_COEFFS := $(MODULE_NAME)/coeffs.csv

$(LINEAR_MODELS_COEFFS): results/single_page_size/mean.csv 
	$(BUILD_LINEAR_MODELS_COEFFS) --mean_file=$< --output=$@

DELETE_TARGETS := $(addsuffix /delete,$(LINEAR_MODELS_COEFFS))
$(MODULE_NAME)/clean: $(DELETE_TARGETS)

include $(ROOT_DIR)/common.mk

