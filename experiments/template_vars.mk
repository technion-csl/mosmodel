DEFAULT_NUM_LAYOUTS ?= 9
DEFAULT_NUM_OF_REPEATS ?= 3
NUMBER_OF_THREADS ?= $(NUMBER_OF_CORES_PER_SOCKET)

ifndef NUM_LAYOUTS
NUM_LAYOUTS := $(DEFAULT_NUM_LAYOUTS)
endif # ifndef NUM_LAYOUTS

ifndef LAYOUTS
LAYOUTS := $(shell seq 1 $(NUM_LAYOUTS))
LAYOUTS := $(addprefix layout,$(LAYOUTS)) 
endif #ifndef LAYOUTS

ifndef NUM_OF_REPEATS
NUM_OF_REPEATS := $(DEFAULT_NUM_OF_REPEATS)
endif # ifndef NUM_OF_REPEATS

ifndef NUMBER_OF_THREADS
NUMBER_OF_THREADS ?= NUMBER_OF_CORES_PER_SOCKET
endif

EXPERIMENT_DIR := $(MODULE_NAME)

$(EXPERIMENT_DIR)/%: NUM_LAYOUTS := $(NUM_LAYOUTS)
$(EXPERIMENT_DIR)/%: NUM_OF_REPEATS := $(NUM_OF_REPEATS)
$(EXPERIMENT_DIR)/%: EXTRA_ARGS_FOR_MOSALLOC := $(EXTRA_ARGS_FOR_MOSALLOC)

ANALYSIS_DIR := $(EXPERIMENT_DIR:experiments%=analysis%)
# ANALYSIS_DIR := $(subst experiments,analysis,$(EXPERIMENT_DIR))
$(ANALYSIS_DIR)/%: NUM_LAYOUTS := $(NUM_LAYOUTS)
$(ANALYSIS_DIR)/%: NUM_OF_REPEATS := $(NUM_OF_REPEATS)

RESULT_DIR := $(subst experiments,results,$(EXPERIMENT_DIR))
RESULT_DIRS += $(RESULT_DIR)

LAYOUTS_DIR := $(EXPERIMENT_DIR)/layouts
LAYOUT_FILES := $(addprefix $(LAYOUTS_DIR)/,$(LAYOUTS))
LAYOUT_FILES := $(addsuffix .csv,$(LAYOUT_FILES))

$(LAYOUTS_DIR): $(LAYOUT_FILES)

EXPERIMENTS := $(addprefix $(EXPERIMENT_DIR)/,$(LAYOUTS)) 
MEAN_RESULTS := $(addsuffix /mean.csv,$(RESULT_DIR))
RESULT_FILES := median.csv std.csv all_repeats.csv
ALL_RESULTS := $(foreach f,$(RESULT_FILES),$(addsuffix /$(f),$(RESULT_DIR)))

REPEATS := $(shell seq 1 $(NUM_OF_REPEATS))
REPEATS := $(addprefix repeat,$(REPEATS)) 

EXPERIMENT_REPEATS := $(foreach experiment,$(EXPERIMENTS),$(foreach repeat,$(REPEATS),$(experiment)/$(repeat)))
MEASUREMENTS := $(addsuffix /perf.out,$(EXPERIMENT_REPEATS))

$(EXPERIMENT_DIR): $(MEASUREMENTS)
$(EXPERIMENTS): $(EXPERIMENT_DIR)/layout%: $(foreach repeat,$(REPEATS),$(addsuffix /$(repeat)/perf.out,$(EXPERIMENT_DIR)/layout%))
$(EXPERIMENT_REPEATS): %: %/perf.out

results: $(RESULT_DIR)
$(RESULT_DIR) $(ALL_RESULTS): $(MEAN_RESULTS)

$(MEAN_RESULTS): LAYOUT_LIST := $(call array_to_comma_separated,$(LAYOUTS))
$(MEAN_RESULTS): NUM_OF_REPEATS := $(NUM_OF_REPEATS)
$(MEAN_RESULTS): results/%/mean.csv: experiments/%
	mkdir -p $(dir $@)
	$(COLLECT_RESULTS) --experiments_root=$< --repeats=$(NUM_OF_REPEATS) \
		--layouts=$(LAYOUT_LIST) --output_dir=$(dir $@) --skip_outliers

DELETED_TARGETS := $(EXPERIMENTS) $(EXPERIMENT_REPEATS) $(LAYOUTS_DIR)
CLEAN_TARGETS := $(addsuffix /clean,$(DELETED_TARGETS))
$(CLEAN_TARGETS): %/clean: %/delete
$(EXPERIMENT_DIR)/clean: $(CLEAN_TARGETS)


