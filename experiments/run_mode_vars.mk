# experiments/run_mode_vars.mk
# RUN_MODE is expected to be set before including module.mk
RUN_MODE ?= st

# Default threads (ST): keep current behavior
ST_NUMBER_OF_THREADS ?= $(NUMBER_OF_CORES_PER_SOCKET)

# Default threads (SMT): 1 thread per workload
SMT_THREADS1 ?= 1
SMT_THREADS2 ?= 1

ifeq ($(RUN_MODE),smt)
  export EXPERIMENTS_TEMPLATE := $(EXPERIMENTS_ROOT)/template_smt.mk
  NUMBER_OF_THREADS ?= $(SMT_THREADS1)
  EXPERIMENTS_WARMUP_DIRS := $(EXPERIMENTS_RUN_DIR)/1/warmup $(EXPERIMENTS_RUN_DIR)/2/warmup
else
  export EXPERIMENTS_TEMPLATE := $(EXPERIMENTS_ROOT)/template.mk
  NUMBER_OF_THREADS ?= $(ST_NUMBER_OF_THREADS)
  EXPERIMENTS_WARMUP_DIRS := $(EXPERIMENTS_RUN_DIR)/warmup
endif

# Keep current OpenMP export mechanism, but now it becomes mode-aware
export OMP_NUM_THREADS := $(NUMBER_OF_THREADS)
export OMP_THREAD_LIMIT := $(OMP_NUM_THREADS)

