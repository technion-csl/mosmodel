MODULE_NAME := experiments/static_auto_mosalloc

STATIC_AUTO_MOSALLOC_NUM_OF_REPEATS := $(NUMBER_OF_SOCKETS)
NUM_LAYOUTS := 55
NUM_OF_REPEATS := $(STATIC_AUTO_MOSALLOC_NUM_OF_REPEATS)
undefine LAYOUTS #allow the template to create new layouts based on the new NUM_LAYOUTS

include $(EXPERIMENTS_TEMPLATE)

CREATE_STATIC_AUTO_MOSALLOC_LAYOUTS_SCRIPT := $(MODULE_NAME)/createLayouts.py
$(LAYOUT_FILES): $(MEMORY_FOOTPRINT_FILE) analysis/pebs_tlb_miss_trace/mem_bins_2mb.csv
	$(CREATE_STATIC_AUTO_MOSALLOC_LAYOUTS_SCRIPT) \
		--memory_footprint=$(MEMORY_FOOTPRINT_FILE) \
		--input_file=$(MEM_BINS_2MB_CSV_FILE) \
		--num_layouts=$(NUM_LAYOUTS) \
		--output=$(dir $@)/..

override undefine NUM_LAYOUTS
override undefine NUM_OF_REPEATS
override undefine LAYOUTS