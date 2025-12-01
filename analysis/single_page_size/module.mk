MODULE_NAME := analysis/single_page_size
SUBMODULES := 

LAYOUTS := layout2mb layout4kb layout1gb

include $(COMMON_ANALYSIS_MAKEFILE)

undefine LAYOUTS

