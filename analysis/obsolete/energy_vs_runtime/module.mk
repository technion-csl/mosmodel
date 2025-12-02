MODULE_NAME := analysis/energy_vs_runtime
SUBMODULES := 

EXPERIMENTS_ROOT := experiments/growing_window_2m

$(MODULE_NAME)/%: NUM_OF_REPEATS := $(GROWING_WINDOW_NUM_OF_REPEATS)
$(MODULE_NAME)/%: CONFIGURATION_LIST := \
	$(call array_to_comma_separated,$(GROWING_WINDOW_CONFIGURATIONS))

RESULT_DIRS := $(addprefix $(MODULE_NAME)/,$(INTERESTING_BENCHMARKS))
MEAN_FILES := $(addsuffix /mean.csv,$(RESULT_DIRS))
ENERGY_SCATTER_FILES := $(addsuffix /energy_scatter.csv,$(RESULT_DIRS))
RUNTIME_SCATTER_FILES := $(addsuffix /runtime_scatter.csv,$(RESULT_DIRS))
SCATTER_CHARTS := $(addsuffix /scatter.pdf,$(RESULT_DIRS))

$(MODULE_NAME): $(SCATTER_CHARTS)

PER_BENCHMARK_TARGETS := $(addprefix $(MODULE_NAME)/,$(INTERESTING_BENCHMARKS))
$(PER_BENCHMARK_TARGETS): %: %/scatter.pdf

$(SCATTER_CHARTS): %/scatter.pdf: %/energy_scatter.csv %/runtime_scatter.csv
	gnuplot -e "input_file='$^'" -e "output_file='$@'" $(SCATTER_PLOT)

$(ENERGY_SCATTER_FILES): %/energy_scatter.csv: %/mean.csv
	$(ARRANGE_DATA_TO_PLOT) --normalize='separate' \
		--y-metric='power/energy-pkg/' --mean_file=$< --output=$@

$(RUNTIME_SCATTER_FILES): %/runtime_scatter.csv: %/mean.csv
	$(ARRANGE_DATA_TO_PLOT) --normalize='by-y' \
		--y-metric='cpu-cycles' --mean_file=$< --output=$@

$(MEAN_FILES): $(MODULE_NAME)/%/mean.csv: | $(EXPERIMENTS_ROOT)/%
	$(COLLECT_RESULTS) --experiments_root=$(EXPERIMENTS_ROOT) \
		--configurations=$(CONFIGURATION_LIST) --benchmarks=$* \
		--repeats=$(NUM_OF_REPEATS) --output_dir=$(dir $@)

DELETE_TARGETS := $(addsuffix /delete,$(RESULT_DIRS))
$(MODULE_NAME)/clean: $(DELETE_TARGETS)

include $(ROOT_DIR)/common.mk

