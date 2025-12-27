MODULE_NAME := experiments/subgroups_windows

NUM_LAYOUTS := 9
undefine LAYOUTS #allow the template to create new layouts based on the new NUM_LAYOUTS

include $(EXPERIMENTS_TEMPLATE)

CREATE_SUBGROUPS_WINDOWS_LAYOUTS_SCRIPT := $(MODULE_NAME)/createLayouts.py
$(LAYOUT_FILES): $(MEMORY_FOOTPRINT_FILE) $(HOT_REGION_FILE)
	$(CREATE_SUBGROUPS_WINDOWS_LAYOUTS_SCRIPT) \
		--memory_footprint=$(MEMORY_FOOTPRINT_FILE) \
		--pebs_mem_bins=$(MEM_BINS_2MB_CSV_FILE) \
		--exp_dir=$(dir $@)/..

override undefine NUM_LAYOUTS
override undefine LAYOUTS
