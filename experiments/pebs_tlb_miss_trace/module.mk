MODULE_NAME := experiments/pebs_tlb_miss_trace
SUBMODULES :=

PERF_RECORD_FREQUENCY ?= $$(( 2**6 ))
STLB_MISS_LOADS_PEBS_EVENT = $(shell perf list | grep retired | grep mem | grep stlb_miss_loads | tr -d ' ')
STLB_MISS_STORES_PEBS_EVENT = $(shell perf list | grep retired | grep mem | grep stlb_miss_stores | tr -d ' ')
PERF_MEM_STLB_MISSES_EVENTS = $(STLB_MISS_LOADS_PEBS_EVENT):p,$(STLB_MISS_STORES_PEBS_EVENT):p
PERF_MEM_RECORD_CMD = perf record --data --count=$(PERF_RECORD_FREQUENCY) --event=$(PERF_MEM_STLB_MISSES_EVENTS)

PEBS_EXP_DIR := $(MODULE_NAME)
PEBS_EXP_OUT_DIR := $(MODULE_NAME)/repeat1
PEBS_TLB_MISS_TRACE_OUTPUT := $(PEBS_EXP_OUT_DIR)/perf.data

# For PEBS experiments, perf needs to wrap the entire command, so we can't use SET_TASK_AFFINITY_CMD in the prefix
ifdef CPU_ISOLATION
PEBS_PREFIX = $(PERF_MEM_RECORD_CMD) --output=$(ROOT_DIR)/$(PEBS_TLB_MISS_TRACE_OUTPUT) --cpu $(ISOLATED_CPUS) -- $(SET_TASK_AFFINITY_CMD)
else # CSET_SHIELD_RUN and SERIAL_RUN
PEBS_PREFIX = $(SET_TASK_AFFINITY_CMD) $(PERF_MEM_RECORD_CMD) --output=$(ROOT_DIR)/$(PEBS_TLB_MISS_TRACE_OUTPUT) -- $(SET_TASK_AFFINITY_CMD)
endif 

$(PEBS_EXP_OUT_DIR): $(PEBS_TLB_MISS_TRACE_OUTPUT)
$(MODULE_NAME): $(PEBS_TLB_MISS_TRACE_OUTPUT)

$(PEBS_TLB_MISS_TRACE_OUTPUT): experiments/single_page_size/layouts/layout4kb.csv | experiments-prerequisites 
	$(RUN_BENCHMARK) --force \
		--prefix="$(PEBS_PREFIX)" \
		--num_threads=$(NUMBER_OF_THREADS) \
		--num_repeats=1 \
		--exclude_files=$(notdir $@) \
		--submit_command="$(RUN_MOSALLOC_TOOL) --analyze -cpf $(ROOT_DIR)/experiments/single_page_size/layouts/layout4kb.csv --library $(MOSALLOC_TOOL)" \
		--benchmark_dir=$(BENCHMARK_PATH) \
		--output_dir=$(PEBS_EXP_DIR) \
		--run_dir=$(EXPERIMENTS_RUN_DIR)
ifdef CSET_SHIELD_RUN
	sudo chown -R $(USER):$(shell id -gn) $(PEBS_EXP_OUT_DIR)
	sudo chmod -R u+rw $(PEBS_EXP_OUT_DIR)
endif

DELETE_TARGETS := $(addsuffix /delete,$(PEBS_TLB_MISS_TRACE_OUTPUT))

$(MODULE_NAME)/clean:
	rm -rf $(PEBS_EXP_OUT_DIR)


