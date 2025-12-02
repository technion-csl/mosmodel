MODULE_NAME := experiments/growing_window_2m

include $(EXPERIMENTS_TEMPLATE)

CREATE_GROWING_WINDOW_LAYOUTS := $(MODULE_NAME)/createLayouts.py
$(LAYOUT_FILES): $(MEMORY_FOOTPRINT_FILE)
	$(CREATE_GROWING_WINDOW_LAYOUTS) \
		--memory_footprint=$(MEMORY_FOOTPRINT_FILE) --num_layouts=$(NUM_LAYOUTS) \
		--output=$(dir $@)/..


