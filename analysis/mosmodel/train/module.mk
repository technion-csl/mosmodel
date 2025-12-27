MODULE_NAME := analysis/mosmodel/train
SUBMODULES := 

MODEL_EXPERIMENTS := growing_window_2m sliding_window/window_20 sliding_window/window_40 sliding_window/window_60 sliding_window/window_80

include $(MOSMODEL_TEMPLATE_MAKEFILE)
