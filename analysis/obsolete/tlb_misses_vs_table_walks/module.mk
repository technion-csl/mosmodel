MODULE_NAME := analysis/tlb_misses_vs_table_walks
SUBMODULES := 

RESULTS_ROOT := analysis/unified_windows

$(MODULE_NAME)/%: NUM_OF_REPEATS := $(GROWING_WINDOW_NUM_OF_REPEATS)
$(MODULE_NAME)/%: CONFIGURATION_LIST := \
	$(call array_to_comma_separated,$(GROWING_WINDOW_CONFIGURATIONS))

RESULT_DIRS := $(addprefix $(MODULE_NAME)/,$(INTERESTING_BENCHMARKS))
MEAN_FILES := $(addsuffix /mean.csv,$(RESULT_DIRS))
TLB_MISSES_SCATTER_FILES := $(addsuffix /tlb_misses_scatter.csv,$(RESULT_DIRS))
TABLE_WALKS_SCATTER_FILES := $(addsuffix /table_walks_scatter.csv,$(RESULT_DIRS))
SCATTER_CHARTS := $(addsuffix /scatter.pdf,$(RESULT_DIRS))

$(MODULE_NAME): $(SCATTER_CHARTS)

PER_BENCHMARK_TARGETS := $(addprefix $(MODULE_NAME)/,$(INTERESTING_BENCHMARKS))
$(PER_BENCHMARK_TARGETS): %: %/scatter.pdf

#$(SCATTER_CHARTS): %/scatter.pdf: %/tlb_misses_scatter.csv %/table_walks_scatter.csv
$(SCATTER_CHARTS): %/scatter.pdf: %/tlb_misses_scatter.csv
	gnuplot -e "input_file='$^'" -e "output_file='$@'" -e "size_ratio=1" -e "x_label='tlb misses'" $(SCATTER_PLOT)

$(TLB_MISSES_SCATTER_FILES): %/tlb_misses_scatter.csv: %/mean.csv
	$(ARRANGE_DATA_TO_PLOT)  \
		--x-metric='tlb_misses' --mean_file=$< --output=$@

$(TABLE_WALKS_SCATTER_FILES): %/table_walks_scatter.csv: %/mean.csv
	$(ARRANGE_DATA_TO_PLOT) --normalize='by-y' \
		--x-metric='walk_cycles' --mean_file=$< --output=$@

$(MEAN_FILES): $(MODULE_NAME)/%/mean.csv: $(RESULTS_ROOT)/%/mean.csv
	mkdir -p `dirname $@`
	cp $< $@

DELETE_TARGETS := $(addsuffix /delete,$(RESULT_DIRS))
$(MODULE_NAME)/clean: $(DELETE_TARGETS)

include $(ROOT_DIR)/common.mk

