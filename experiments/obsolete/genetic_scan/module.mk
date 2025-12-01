MODULE_NAME := experiments/genetic_scan
GENETIC_SCAN_EXP_DIR := $(MODULE_NAME)

NUM_LAYOUTS := 33
undefine LAYOUTS #allow the template to create new layouts based on the new NUM_LAYOUTS

include $(EXPERIMENTS_TEMPLATE)

CREATE_GENETIC_SCAN_LAYOUTS := $(MODULE_NAME)/createLayouts.py
$(LAYOUT_FILES): $(GENETIC_SCAN_EXP_DIR)/layouts/%.csv: $(MEMORY_FOOTPRINT_FILE) experiments/single_page_size/layout4kb experiments/single_page_size/layout2mb
	mkdir -p results/single_page_size
	$(COLLECT_RESULTS) --experiments_root=experiments/single_page_size --repeats=$(NUM_OF_REPEATS) \
		--layouts=layout4kb,layout2mb --output_dir=results/single_page_size
	mkdir -p results/genetic_scan
	layouts=''
	-layouts=`ls -dm experiments/genetic_scan/layout*[!s] | sed 's,experiments/genetic_scan/,,g' | tr -d ' \n'`
	$(COLLECT_RESULTS) --experiments_root=experiments/genetic_scan --repeats=$(NUM_OF_REPEATS) \
		--layouts=$$layouts --output_dir=results/genetic_scan
	$(CREATE_GENETIC_SCAN_LAYOUTS) \
		--memory_footprint=$(MEMORY_FOOTPRINT_FILE) \
		--single_page_size_results=results/single_page_size/mean.csv \
		--genetic_results=results/genetic_scan/mean.csv \
		--layout=$* \
		--layouts_dir=$(dir $@)/..

override undefine NUM_LAYOUTS
override undefine LAYOUTS
