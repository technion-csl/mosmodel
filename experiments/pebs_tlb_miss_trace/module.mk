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
$(info ======== PEBS DEBUG ========)
$(info CPU_ISOLATION = [$(CPU_ISOLATION)])
$(info CSET_SHIELD_RUN = [$(CSET_SHIELD_RUN)])
$(info ISOLATED_CPUS = [$(ISOLATED_CPUS)])
$(info SET_TASK_AFFINITY_CMD = [$(SET_TASK_AFFINITY_CMD)])
$(info ============================)
# ifdef CSET_SHIELD_RUN
# $(info >>> Taking CSET_SHIELD_RUN branch)
# # cset shield wraps perf, which then wraps the benchmark
# PEBS_PREFIX = $(SET_TASK_AFFINITY_CMD) $(PERF_MEM_RECORD_CMD) --output=$(ROOT_DIR)/$(PEBS_TLB_MISS_TRACE_OUTPUT) -- $(SET_TASK_AFFINITY_CMD)
# else ifdef CPU_ISOLATION
# $(info >>> Taking CPU_ISOLATION branch)
# PEBS_PREFIX = $(PERF_MEM_RECORD_CMD) --output=$(ROOT_DIR)/$(PEBS_TLB_MISS_TRACE_OUTPUT) --cpu $(ISOLATED_CPUS) -- $(SET_TASK_AFFINITY_CMD)
# else
$(info >>> Taking SERIAL_RUN branch (no isolation))
# SERIAL_RUN - no CPU affinity, just perf record
include $(MODULE_NAME)/run_mode.mk


# ifdef CSET_SHIELD_RUN
# 	sudo chown -R $(USER):$(shell id -gn) $(PEBS_EXP_OUT_DIR)
# 	sudo chmod -R u+rw $(PEBS_EXP_OUT_DIR)
# endif

DELETE_TARGETS := $(addsuffix /delete,$(PEBS_TLB_MISS_TRACE_OUTPUT))

$(MODULE_NAME)/clean:
	rm -rf $(PEBS_EXP_OUT_DIR)


