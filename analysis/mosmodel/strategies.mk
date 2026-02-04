# List of strategies to compare (parent can override)
MOSMODEL_STRATEGIES ?= \
	growing_window_2m \
	moselect

# Default common test generator(s) (parent can override)
MOSMODEL_DEFAULT_TEST_LAYOUT_GENERATORS ?= random_window_2m
