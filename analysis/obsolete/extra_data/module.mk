MODULE_NAME := analysis/extra_data
SUBMODULES := 

CONCAT_RESULT_FILES_SCRIPT := $(ROOT_DIR)/$(MODULE_NAME)/concatResultFiles.py
EXTRA_DATA_MEAN_FILE := $(MODULE_NAME)/mean.csv

$(MODULE_NAME): $(EXTRA_DATA_MEAN_FILE)

MODEL_EXPERIMENTS := growing_window_2m sliding_window/window_20 sliding_window/window_40 sliding_window/window_60 sliding_window/window_80 random_window_2m extra_random_2m

RESULT_MEAN_FILES := $(addprefix results/,$(MODEL_EXPERIMENTS))
RESULT_MEAN_FILES := $(addsuffix /mean.csv,$(RESULT_MEAN_FILES))

RESULT_MEAN_FILES_LIST := $(call array_to_comma_separated,$(RESULT_MEAN_FILES)) 

$(EXTRA_DATA_MEAN_FILE): $(RESULT_MEAN_FILES)
	$(CONCAT_RESULT_FILES_SCRIPT) --files=$(RESULT_MEAN_FILES_LIST) --output=$(EXTRA_DATA_MEAN_FILE)

$(MODULE_NAME)/clean:
	rm -f $(EXTRA_DATA_MEAN_FILE)

