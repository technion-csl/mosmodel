MODULE_NAME := slurm
SUBMODULES := 

include $(ROOT_DIR)/common.mk

SLURM_CONFIG_TEMPLATE := $(MODULE_NAME)/slurm.conf.template
SLURM_CONFIG_FILE := $(MODULE_NAME)/slurm.conf
SLURM_INSTALLED_CONFIG_FILE := /etc/slurm-llnl/slurm.conf
SLURMCTLD := /usr/sbin/slurmctld
SLURM_SPOOL_DIR := /var/spool/slurm
SLURM_APT_PACKAGES := slurm-wlm slurm-wlm-basic-plugins

$(MODULE_NAME): $(SLURM_INSTALLED_CONFIG_FILE)
	if [[ `scontrol ping` =~ "DOWN/DOWN" ]] ; then
		sudo slurmctld
		sudo slurmd
	fi

$(SLURM_INSTALLED_CONFIG_FILE): $(SLURM_CONFIG_FILE)
	sudo cp -f $< $@

$(SLURM_CONFIG_FILE): $(SLURM_CONFIG_TEMPLATE) $(SLURMCTLD)
	hostname=$$(hostname)
	sockets=$$(lscpu | grep "NUMA node(s)" | cut -d ':' -f 2 | tr -d ' ')
	cores=$$(lscpu | grep "Core(s) per socket" | cut -d ':' -f 2 | tr -d ' ')
	threads=$$(lscpu | grep "Thread(s) per core" | cut -d ':' -f 2 | tr -d ' ')
	sed "s,HOSTNAME,$$hostname,g" $< > $@
	sed -i "s,SOCKETS,$$sockets,g" $@
	sed -i "s,CORES,$$cores,g" $@
	sed -i "s,THREADS,$$threads,g" $@

$(SLURMCTLD):
	$(APT_INSTALL) $(SLURM_APT_PACKAGES)
	sudo mkdir -p $(SLURM_SPOOL_DIR)
	sudo chown slurm:slurm $(SLURM_SPOOL_DIR)

$(MODULE_NAME)/clean:
	sudo apt remove --purge $(SLURM_APT_PACKAGES)
	rm -rf $(SLURM_CONFIG_FILE)
	sudo rm -rf $(SLURM_SPOOL_DIR)

