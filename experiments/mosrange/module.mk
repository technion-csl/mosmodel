MOSRANGE_EXPERIMENT_NAME := mosrange
MODULE_NAME := experiments/$(MOSRANGE_EXPERIMENT_NAME)

# workaround to skip removing the first measurement (layout1/repeat1/perf.out) 
# when $(MEASUREMTENTS) target fails; MEASUREMENTS variable will contain now 
# the layout results in reverse order.
ifdef DEFAULT_NUM_LAYOUTS
NUM_LAYOUTS := $(DEFAULT_NUM_LAYOUTS)
endif
LAYOUTS := $(shell seq 1 $(NUM_LAYOUTS) | tac)
LAYOUTS := $(addprefix layout,$(LAYOUTS)) 

NUM_OF_REPEATS := $(DEFAULT_NUM_OF_REPEATS)

include $(EXPERIMENTS_VARS_TEMPLATE)

MOSRANGE_METRIC_NAME ?= stlb_misses
ifdef MOSRANGE_METRIC_COVERAGE
MOSRANGE_COVERAGE_ARG := --metric_coverage=$(MOSRANGE_METRIC_COVERAGE)
endif
ifdef MOSRANGE_METRIC_VALUE
MOSRANGE_VALUE_ARG := --metric_value=$(MOSRANGE_METRIC_VALUE)
endif

MOSRANGE_EXPERIMENT_ROOT_DIR := $(MODULE_NAME)
MOSRANGE_RESULTS_DIR := results/$(MOSRANGE_EXPERIMENT_NAME)

PEBS_FILE := analysis/pebs_tlb_miss_trace/mem_bins_2mb.csv
MOSRANGE_RUN_BENCHMARK := $(MODULE_NAME)/run_benchmark.sh
RUN_MOSRANGE_EXP_SCRIPT := $(MODULE_NAME)/runExperiment.py

$(MOSRANGE_RUN_BENCHMARK): $(CUSTOM_RUN_EXPERIMENT_SCRIPT).$(MOSRANGE_EXPERIMENT_NAME)
	cp $< $@
$(CUSTOM_RUN_EXPERIMENT_SCRIPT).$(MOSRANGE_EXPERIMENT_NAME): EXPERIMENT_NAME := $(MOSRANGE_EXPERIMENT_NAME)
$(CUSTOM_RUN_EXPERIMENT_SCRIPT).$(MOSRANGE_EXPERIMENT_NAME): NUM_OF_REPEATS := $(NUM_OF_REPEATS)
$(CUSTOM_RUN_EXPERIMENT_SCRIPT).$(MOSRANGE_EXPERIMENT_NAME): | $(CUSTOM_RUN_EXPERIMENT_SCRIPT)
	cp $| $@

$(MEASUREMENTS): $(MEMORY_FOOTPRINT_FILE) $(PEBS_FILE) $(MOSRANGE_RUN_BENCHMARK)
	$(RUN_MOSRANGE_EXP_SCRIPT) \
		--memory_footprint=$(MEMORY_FOOTPRINT_FILE) \
		--pebs_mem_bins=$(MEM_BINS_2MB_CSV_FILE) \
		--exp_root_dir=$(MOSRANGE_EXPERIMENT_ROOT_DIR) \
		--results_dir=$(MOSRANGE_RESULTS_DIR) \
		--run_experiment_cmd=$(MOSRANGE_RUN_BENCHMARK) \
		--num_layouts=$(NUM_LAYOUTS) \
		--num_repeats=$(NUM_OF_REPEATS) \
		--metric=$(MOSRANGE_METRIC_NAME) \
		$(MOSRANGE_VALUE_ARG) $(MOSRANGE_COVERAGE_ARG) $(MOSRANGE_EXTRA_ARGS)

MEDIAN_RESULTS := $(addsuffix /median.csv,$(RESULT_DIR))
$(MEDIAN_RESULTS): results/%/median.csv: experiments/%
	mkdir -p $(dir $@)
	$(COLLECT_RESULTS) --experiments_root=$< --repeats=$(NUM_OF_REPEATS) \
		--output_dir=$(dir $@) --skip_outliers

.PHONY: $(MODULE_NAME)
