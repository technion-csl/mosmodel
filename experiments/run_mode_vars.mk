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
else
  export EXPERIMENTS_TEMPLATE := $(EXPERIMENTS_ROOT)/template.mk
  NUMBER_OF_THREADS ?= $(ST_NUMBER_OF_THREADS)
endif

