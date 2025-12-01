MODULE_NAME := analysis/mmap_brk_effects
SUBMODULES := 

include $(ROOT_DIR)/common.mk

MMAP_4K_BRK_4K_CSV_FILE := $(MODULE_NAME)/mmap_4k_brk_4k/mean.csv
MMAP_1G_BRK_1G_CSV_FILE := $(MODULE_NAME)/mmap_1g_brk_1g/mean.csv
MMAP_1G_BRK_4K_CSV_FILE := $(MODULE_NAME)/mmap_1g_brk_4k/mean.csv
MMAP_4K_BRK_1G_CSV_FILE := $(MODULE_NAME)/mmap_4k_brk_1g/mean.csv

CALCULATE_EFFECTS := $(ROOT_DIR)/$(MODULE_NAME)/calculateEffects.py
ARRANGE_RESPONSES := $(ROOT_DIR)/$(MODULE_NAME)/arrangeResponses.py

CSV_FILES := $(addprefix $(MODULE_NAME)/,$(MMAP_VS_BRK_SUBMODULES))
CSV_FILES := $(addsuffix /mean.csv,$(CSV_FILES))
RESPONSES_FILE := $(MODULE_NAME)/responses.csv
EFFECTS_FILE := $(MODULE_NAME)/effects.csv
SUMMARY_FILES := $(CSV_FILES) $(RESPONSES_FILE) $(EFFECTS_FILE)

$(MODULE_NAME): $(SUMMARY_FILES)

$(EFFECTS_FILE): $(RESPONSES_FILE)
	$(CALCULATE_EFFECTS) $| --output=$@ --responses=$(RESPONSES_FILE)

$(RESPONSES_FILE): $(CSV_FILES)
	$(ARRANGE_RESPONSES) $| --output=$@ \
		--response00=$(MMAP_4K_BRK_4K_CSV_FILE) \
		--response11=$(MMAP_1G_BRK_1G_CSV_FILE) \
		--response10=$(MMAP_1G_BRK_4K_CSV_FILE) \
		--response01=$(MMAP_4K_BRK_1G_CSV_FILE)

$(CSV_FILES): $(MODULE_NAME)/%/mean.csv: | experiments/mmap_vs_brk/%
	$(COLLECT_RESULTS) --experiments_root=$(dir $|) \
		--layout=$* --repeats 1 --output_dir=$(dir $@)

DELETE_TARGETS := $(addsuffix /delete,$(SUMMARY_FILES))
$(MODULE_NAME)/clean: $(DELETE_TARGETS)

