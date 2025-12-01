MODULE_NAME := experiments/extra_random_2m

EXTRA_EXPS_NUM_LAYOUTS := 46
NUM_LAYOUTS := $(EXTRA_EXPS_NUM_LAYOUTS)
undefine LAYOUTS #allow the template to create new layouts based on the new NUM_LAYOUTS

include $(EXPERIMENTS_TEMPLATE)

CREATE_RANDOM_WINDOW_LAYOUTS := $(MODULE_NAME)/createLayouts.py
$(LAYOUT_FILES): $(MEMORY_FOOTPRINT_FILE)
	$(CREATE_RANDOM_WINDOW_LAYOUTS) \
		--memory_footprint=$(MEMORY_FOOTPRINT_FILE) --num_layouts=$(EXTRA_EXPS_NUM_LAYOUTS) \
		--output=$(dir $@)/..

override undefine NUM_LAYOUTS
override undefine LAYOUTS

