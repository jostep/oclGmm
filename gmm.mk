# Printing flags
USE_PRINT_BUFFER = 0
PRINT_LEVEL = 4

GMM_CONFIGS :=

# Collect runtime statistics?
GMM_CONFIGS += -DGMM_CONFIG_STATS

# Enable RW hint
#GMM_CONFIGS += -DGMM_CONFIG_RW

# Copy on write, including memset flag
GMM_CONFIGS += -DGMM_CONFIG_COW

# Use asynchronous DMA
GMM_CONFIGS += -DGMM_CONFIG_DMA_ASYNC

# Use the radical version of HtoD?
#GMM_CONFIGS += -DGMM_CONFIG_HTOD_RADICAL

# Replacement policy
GMM_CONFIGS += -DGMM_REPLACEMENT_LRU
#GMM_CONFIGS += -DGMM_REPLACEMENT_LFU
#GMM_CONFIGS += -DGMM_REPLACEMENT_RANDOM
