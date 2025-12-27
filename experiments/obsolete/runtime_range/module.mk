MODULE_NAME := experiments/runtime_range
RUNTIME_RANGE_EXPERIMENT := $(MODULE_NAME)

RUNTIME_RANGE_NUM_OF_REPEATS := $(NUMBER_OF_SOCKETS)
NUM_LAYOUTS := 55
NUM_OF_REPEATS := $(RUNTIME_RANGE_NUM_OF_REPEATS)
undefine LAYOUTS #allow the template to create new layouts based on the new NUM_LAYOUTS

include $(EXPERIMENTS_TEMPLATE)

RUNTIME_RANGE_WALK_CYCLES ?= 4e12
CREATE_RUNTIME_RANGE_LAYOUTS := $(MODULE_NAME)/createLayouts.py
$(LAYOUT_FILES): $(MEMORY_FOOTPRINT_FILE) analysis/pebs_tlb_miss_trace/mem_bins_2mb.csv results/single_page_size/mean.csv
	$(CREATE_RUNTIME_RANGE_LAYOUTS) \
		--memory_footprint=$(MEMORY_FOOTPRINT_FILE) \
		--single_page_size_mean=results/single_page_size/mean.csv \
		--pebs_mem_bins=$(MEM_BINS_2MB_CSV_FILE) \
		--walk_cycles=$(RUNTIME_RANGE_WALK_CYCLES) \
		--num_layouts=$(NUM_LAYOUTS) \
		--layouts_dir=$(dir $@)/..

override undefine NUM_LAYOUTS
override undefine NUM_OF_REPEATS
override undefine LAYOUTS
