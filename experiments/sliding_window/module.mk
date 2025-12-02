SLIDING_WINDOW_MODULE_NAME := experiments/sliding_window
HOT_REGION_FILE := $(SLIDING_WINDOW_MODULE_NAME)/hot_region.txt
SLIDING_WINDOW_WEIGHTS := 20 40 50 60 80
CREATE_SLIDING_WINDOW_LAYOUTS := $(SLIDING_WINDOW_MODULE_NAME)/createLayouts.py

define sliding-makefiles
SLIDING_WINDOW_WEIGHT := $(1)
include $(SLIDING_WINDOW_MODULE_NAME)/template.mk
endef

$(foreach w,$(SLIDING_WINDOW_WEIGHTS),$(eval $(call sliding-makefiles,$(w))))
