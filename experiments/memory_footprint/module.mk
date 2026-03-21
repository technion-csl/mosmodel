MODULE_NAME := experiments/memory_footprint
LAYOUTS := layout4kb
EXPERIMENT_DIR := $(MODULE_NAME)
EXPERIMENTS := $(EXPERIMENT_DIR)/$(LAYOUTS)
MEASUREMENTS := $(EXPERIMENTS)/repeat0/perf.out
LAYOUT_FILES := $(EXPERIMENT_DIR)/layouts/$(LAYOUTS).csv

EXTRA_ARGS_FOR_MOSALLOC := --analyze

$(EXPERIMENT_DIR): $(EXPERIMENTS)

$(EXPERIMENTS): $(MEASUREMENTS)

$(MEASUREMENTS): EXTRA_ARGS_FOR_MOSALLOC := $(EXTRA_ARGS_FOR_MOSALLOC)
$(MEASUREMENTS): %/repeat0/perf.out: $(EXPERIMENT_DIR)/layouts/layout4kb.csv | experiments-prerequisites
	echo ========== [INFO] start producing: $@ ==========
	$(RUN_BENCHMARK) \
		--num_threads=$(NUMBER_OF_THREADS) \
		--repeat=repeat0 \
		--submit_command \
		"$(SET_CPU_MEMORY_AFFINITY) $(BOUND_MEMORY_NODE) $(MEASURE_GENERAL_METRICS)  \
		$(RUN_MOSALLOC_TOOL) --library $(MOSALLOC_TOOL) -cpf $(ROOT_DIR)/$< $(EXTRA_ARGS_FOR_MOSALLOC) --" \
		--benchmark_dir=$(BENCHMARK1) \
		--output_dir=$* \
		--run_dir=$(EXPERIMENTS_RUN_DIR)/memory_footprint/repeat0/1

CREATE_MEMORY_FOOTPRINT_LAYOUTS := $(MODULE_NAME)/createLayouts.py
$(LAYOUT_FILES):
	ram_size_kb=$(shell grep MemTotal /proc/meminfo | cut -d ':' -f 2 | sed 's, ,,g' | sed 's,kB,,g')
	$(CREATE_MEMORY_FOOTPRINT_LAYOUTS) --mem_max_size_kb=$$ram_size_kb \
		--output=$(dir $@)/..

# undefine LAYOUTS to allow next makefiles to use the defaults LAYOUTS
undefine EXTRA_ARGS_FOR_MOSALLOC
undefine LAYOUTS
