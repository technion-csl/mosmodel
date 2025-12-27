MOSELECT_EXPERIMENT_NAME := moselect
MODULE_NAME := experiments/$(MOSELECT_EXPERIMENT_NAME)

MOSELECT_EXPERIMENT := $(MODULE_NAME)
MOSELECT_RESULTS := $(subst experiments,results,$(MOSELECT_EXPERIMENT))
MOSELECT_NUM_OF_REPEATS ?= 2
MOSELECT_NUM_LAYOUTS ?= 9
MOSELECT_MAX_GAP ?= 4
NUM_LAYOUTS := $(MOSELECT_NUM_LAYOUTS)
NUM_OF_REPEATS := $(MOSELECT_NUM_OF_REPEATS)
undefine LAYOUTS #allow the template to create new layouts based on the new NUM_LAYOUTS

include $(EXPERIMENTS_TEMPLATE)

CREATE_MOSELECT_LAYOUTS_SCRIPT := $(MODULE_NAME)/createLayouts.py
$(LAYOUT_FILES): $(MOSELECT_EXPERIMENT)/layouts/%.csv: $(MEMORY_FOOTPRINT_FILE) analysis/pebs_tlb_miss_trace/mem_bins_2mb.csv
	mkdir -p results/$(MOSELECT_EXPERIMENT_NAME)
	$(COLLECT_RESULTS) --experiments_root=$(MOSELECT_EXPERIMENT) --repeats=$(NUM_OF_REPEATS) \
		--output_dir=$(MOSELECT_RESULTS) --skip_outliers
	$(CREATE_MOSELECT_LAYOUTS_SCRIPT) \
		--memory_footprint=$(MEMORY_FOOTPRINT_FILE) \
		--pebs_mem_bins=$(MEM_BINS_2MB_CSV_FILE) \
		--layout=$* \
		--max_gap=$(MOSELECT_MAX_GAP) \
		--max_budget=$(MOSELECT_NUM_LAYOUTS) \
		--exp_dir=$(dir $@)/.. \
		--results_file=$(MOSELECT_RESULTS)/median.csv

override undefine NUM_LAYOUTS
override undefine NUM_OF_REPEATS
override undefine LAYOUTS
