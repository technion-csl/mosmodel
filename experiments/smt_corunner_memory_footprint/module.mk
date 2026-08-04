MODULE_NAME := experiments/smt_corunner_memory_footprint
EXPERIMENT_DIR := $(MODULE_NAME)
EXPERIMENT := $(EXPERIMENT_DIR)/layout4kb
MEASUREMENT := $(EXPERIMENT)/repeat0/perf.out
LAYOUT_FILE := $(EXPERIMENT_DIR)/layouts/layout4kb.csv

SMT_CORUNNER_MEMORY_FOOTPRINT_FILE := experiments/smt_corunner_memory_footprint.csv
EXTRA_ARGS_FOR_MOSALLOC := --analyze

$(EXPERIMENT_DIR): $(EXPERIMENT)
$(EXPERIMENT): $(MEASUREMENT)

$(MEASUREMENT): EXTRA_ARGS_FOR_MOSALLOC := $(EXTRA_ARGS_FOR_MOSALLOC)
$(MEASUREMENT): %/repeat0/perf.out: $(LAYOUT_FILE) | experiments-prerequisites
	echo ========== [INFO] start producing SMT co-runner memory footprint: $@ ==========
	$(RUN_BENCHMARK) \
		--num_threads=$(SMT_THREADS2) \
		--repeat=repeat0 \
		--submit_command \
		"$(SET_CPU_MEMORY_AFFINITY) $(BOUND_MEMORY_NODE) $(CPU_MEMORY_AFFINITY_ARGS2) $(MEASURE_GENERAL_METRICS) \
		$(RUN_MOSALLOC_TOOL) --library $(MOSALLOC_TOOL) -cpf $(ROOT_DIR)/$< $(EXTRA_ARGS_FOR_MOSALLOC) --" \
		--benchmark_dir=$(BENCHMARK2) \
		--output_dir=$* \
		--run_dir=$(EXPERIMENTS_RUN_DIR)/smt_corunner_memory_footprint/repeat0/2

CREATE_SMT_CORUNNER_MEMORY_FOOTPRINT_LAYOUT := experiments/memory_footprint/createLayouts.py
$(LAYOUT_FILE):
	ram_size_kb=$(shell grep MemTotal /proc/meminfo | cut -d ':' -f 2 | sed 's, ,,g' | sed 's,kB,,g')
	$(CREATE_SMT_CORUNNER_MEMORY_FOOTPRINT_LAYOUT) --mem_max_size_kb=$$ram_size_kb \
		--output=$(dir $@)/..

$(SMT_CORUNNER_MEMORY_FOOTPRINT_FILE): | $(EXPERIMENT)
	$(COLLECT_MEMORY_FOOTPRINT) $| --output=$@

$(MODULE_NAME)/clean:
	rm -rf $(EXPERIMENT_DIR)
	rm -f $(SMT_CORUNNER_MEMORY_FOOTPRINT_FILE)

undefine EXTRA_ARGS_FOR_MOSALLOC
