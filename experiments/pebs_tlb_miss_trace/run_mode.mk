# experiments/pebs_tlb_miss_trace/run_mode.mk
# Handles ST/SMT rules for PEBS TLB miss trace module.

ifeq ($(RUN_MODE),smt)
  ifeq ($(strip $(BENCHMARK1)),)
    $(error "RUN_MODE=smt but BENCHMARK1 is not set")
  endif
  ifeq ($(strip $(BENCHMARK2)),)
    $(error "RUN_MODE=smt but BENCHMARK2 is not set")
  endif

  SET_TASK_AFFINITY_CMD_1 := $(SET_TASK_AFFINITY_CMD) --smt 1
  SET_TASK_AFFINITY_CMD_2 := $(SET_TASK_AFFINITY_CMD) --smt 2
else
  SET_TASK_AFFINITY_CMD_1 := $(SET_TASK_AFFINITY_CMD)
endif

# PEBS: perf record must wrap the benchmark command; keep output file as perf.data
PEBS_PREFIX_1 := $(SET_TASK_AFFINITY_CMD_1) $(PERF_MEM_RECORD_CMD) --output=$(ROOT_DIR)/$(PEBS_TLB_MISS_TRACE_OUTPUT) --

# Background output directory for benchmark2 (we delete per-repeat after run)
PEBS_BG_OUT_DIR ?= $(EXPERIMENTS_RUN_DIR)/_smt_bg_out/$(notdir $(MODULE_NAME))

# Targets (same as module.mk expects)
$(PEBS_EXP_OUT_DIR): $(PEBS_TLB_MISS_TRACE_OUTPUT)
$(MODULE_NAME): $(PEBS_TLB_MISS_TRACE_OUTPUT)

ifeq ($(RUN_MODE),smt)

$(PEBS_TLB_MISS_TRACE_OUTPUT): experiments/single_page_size/layouts/layout4kb.csv | experiments-prerequisites
	$(RUN_BENCHMARK) --force \
		--prefix="$(SET_TASK_AFFINITY_CMD_2)" \
		--num_threads=$(NUMBER_OF_THREADS) \
		--num_repeats=1 \
		--benchmark_dir=$(BENCHMARK2) \
		--output_dir=$(PEBS_BG_OUT_DIR) \
		--run_dir=$(EXPERIMENTS_RUN_DIR)/2 & \
	\
	$(RUN_BENCHMARK) --force \
		--prefix="$(PEBS_PREFIX_1)" \
		--num_threads=$(NUMBER_OF_THREADS) \
		--num_repeats=1 \
		--exclude_files=$(notdir $@) \
		--submit_command="$(RUN_MOSALLOC_TOOL) --analyze -cpf $(ROOT_DIR)/experiments/single_page_size/layouts/layout4kb.csv --library $(MOSALLOC_TOOL)" \
		--benchmark_dir=$(BENCHMARK1) \
		--output_dir=$(PEBS_EXP_DIR) \
		--run_dir=$(EXPERIMENTS_RUN_DIR)/1; \
	wait; \
	rm -rf "$(PEBS_BG_OUT_DIR)/repeat1"

else  # RUN_MODE=st

$(PEBS_TLB_MISS_TRACE_OUTPUT): experiments/single_page_size/layouts/layout4kb.csv | experiments-prerequisites
	$(RUN_BENCHMARK) --force \
		--prefix="$(PEBS_PREFIX_1)" \
		--num_threads=$(NUMBER_OF_THREADS) \
		--num_repeats=1 \
		--exclude_files=$(notdir $@) \
		--submit_command="$(RUN_MOSALLOC_TOOL) --analyze -cpf $(ROOT_DIR)/experiments/single_page_size/layouts/layout4kb.csv --library $(MOSALLOC_TOOL)" \
		--benchmark_dir=$(BENCHMARK_PATH) \
		--output_dir=$(PEBS_EXP_DIR) \
		--run_dir=$(EXPERIMENTS_RUN_DIR)

endif

