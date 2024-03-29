MODULE_NAME := experiments/pebs_tlb_miss_trace
SUBMODULES :=

PERF_RECORD_FREQUENCY := $$(( 2**15 ))
STLB_MISS_LOADS_PEBS_EVENT = $(shell perf list | grep retired | grep mem | grep stlb_miss_loads | tr -d ' ')
STLB_MISS_STORES_PEBS_EVENT = $(shell perf list | grep retired | grep mem | grep stlb_miss_stores | tr -d ' ')
PERF_MEM_STLB_MISSES_EVENTS = $(STLB_MISS_LOADS_PEBS_EVENT):p,$(STLB_MISS_STORES_PEBS_EVENT):p
PERF_MEM_RECORD_CMD = perf record --data --count=$(PERF_RECORD_FREQUENCY) --event=$(PERF_MEM_STLB_MISSES_EVENTS)

PEBS_EXP_OUT_DIR := $(MODULE_NAME)/repeat0
PEBS_TLB_MISS_TRACE_OUTPUT := $(PEBS_EXP_OUT_DIR)/perf.data
$(PEBS_EXP_OUT_DIR): $(PEBS_TLB_MISS_TRACE_OUTPUT)

$(MODULE_NAME): $(PEBS_TLB_MISS_TRACE_OUTPUT)

$(PEBS_TLB_MISS_TRACE_OUTPUT): experiments/single_page_size/layouts.txt | experiments-prerequisites 
	ARGS_FOR_MOSALLOC="$(shell grep layout4k experiments/single_page_size/layouts.txt | cut -d ':' -f 2)"
	$(RUN_BENCHMARK) --exclude_files=$(notdir $@) \
		--submit_command "$(PERF_MEM_RECORD_CMD) -- $(RUN_MOSALLOC_TOOL) --analyze $$ARGS_FOR_MOSALLOC --library $(MOSALLOC_TOOL)" \
		$(BENCHMARK_PATH) $(dir $@)

DELETE_TARGETS := $(addsuffix /delete,$(PEBS_TLB_MISS_TRACE_OUTPUT))

$(MODULE_NAME)/clean:
	rm -rf $(PEBS_EXP_OUT_DIR)


