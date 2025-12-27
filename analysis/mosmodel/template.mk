# MODEL_EXPERIMENTS should be set before calling this template

R_SQUARES := $(MODULE_NAME)/r_squares.csv
SCATTER_CHART := $(MODULE_NAME)/scatter.pdf
WHISKER_CHART := $(MODULE_NAME)/whisker.pdf
REPEATS_CHART := $(MODULE_NAME)/all_repeats.pdf
SCATTER_CSV_FILE := $(MODULE_NAME)/scatter.csv
WHISKER_CSV_FILE := $(MODULE_NAME)/whisker.csv
REPEATS_CSV_FILE := $(MODULE_NAME)/all_repeats.csv
MEAN_CSV_FILE := $(MODULE_NAME)/mean.csv

MODEL_MEAN_CSV_FILES := $(addsuffix /mean.csv,$(MODEL_EXPERIMENTS))
MODEL_MEAN_CSV_FILES := $(addprefix results/,$(MODEL_MEAN_CSV_FILES))
MODEL_EXPERIMENTS := $(addprefix analysis/,$(MODEL_EXPERIMENTS))
MODEL_SCATTER_CSV_FILES := $(addsuffix /scatter.csv,$(MODEL_EXPERIMENTS))
MODEL_WHISKER_CSV_FILES := $(addsuffix /whisker.csv,$(MODEL_EXPERIMENTS))
MODEL_ALL_REPEATS_CSV_FILES := $(addsuffix /all_repeats_scatter.csv,$(MODEL_EXPERIMENTS))

TARGETS := $(SCATTER_CHART) $(WHISKER_CHART) $(REPEATS_CHART) $(SCATTER_CSV_FILE) $(REPEATS_CSV_FILE) $(MEAN_CSV_FILE) $(R_SQUARES) $(WHISKER_CSV_FILE)

$(MODULE_NAME): $(TARGETS)

$(SCATTER_CHART): $(SCATTER_CSV_FILE) 
	legend=`echo $(MODEL_EXPERIMENTS) | sed -e 's,_window,,g' | sed -e 's,_,-,g'`
	gnuplot -e "input_file='$^'" -e "output_file='$@'" -e "legend='$$legend'" $(SCATTER_PLOT)

$(SCATTER_CSV_FILE): $(MODEL_SCATTER_CSV_FILES) 
	head -n 1 -q $< > $@
	tail -n +2 -q $^ >> $@

$(REPEATS_CHART): $(REPEATS_CSV_FILE)
	gnuplot -e "input_file='$^'" -e "output_file='$@'" $(SCATTER_PLOT)

$(REPEATS_CSV_FILE): $(MODEL_ALL_REPEATS_CSV_FILES)
	head -n 1 -q $< > $@
	tail -n +2 -q $^ >> $@

$(WHISKER_CHART): $(WHISKER_CSV_FILE)
	gnuplot -e "input_file='$^'" -e "output_file='$@'" $(WHISKER_PLOT)

$(WHISKER_CSV_FILE): $(MODEL_WHISKER_CSV_FILES)
	head -n 1 -q $< > $@
	tail -n +2 -q $^ >> $@

$(MEAN_CSV_FILE): $(MODEL_MEAN_CSV_FILES)
	head -n 1 -q $< > $@
	tail -n +2 -q $^ >> $@

$(R_SQUARES): $(MEAN_CSV_FILE)
	$(CALCULATE_R_SQUARES) --input=$< --output=$@

DELETE_TARGETS := $(addsuffix /delete,$(TARGETS))
$(MODULE_NAME)/clean: $(DELETE_TARGETS)

include $(ROOT_DIR)/common.mk

