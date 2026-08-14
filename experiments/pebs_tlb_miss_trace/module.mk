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

# PEBS has its own run-mode setup because perf record wraps the measured command.
include $(MODULE_NAME)/run_mode.mk

DELETE_TARGETS := $(addsuffix /delete,$(PEBS_TLB_MISS_TRACE_OUTPUT))

$(MODULE_NAME)/clean:
	rm -rf $(PEBS_EXP_OUT_DIR)


