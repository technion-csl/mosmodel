# List of strategies to compare
MOSMODEL_STRATEGIES := \
	growing_window_2m \
	sliding_window \
	paper_all \
	moselect

# Training experiments per strategy
TRAIN_EXPERIMENTS_growing_window_2m := growing_window_2m
TRAIN_EXPERIMENTS_random_window_2m := random_window_2m
TRAIN_EXPERIMENTS_sliding_window := sliding_window/window_40
TRAIN_EXPERIMENTS_paper_all := random_window_2m\
	growing_window_2m \
	sliding_window/window_20 sliding_window/window_40 sliding_window/window_60 sliding_window/window_80
TRAIN_EXPERIMENTS_moselect := moselect

