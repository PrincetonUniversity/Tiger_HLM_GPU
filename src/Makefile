# # ─────────────────────────────────────────────────────────────────────────────
# # Makefile for rk45_solver (Model204 only)
# # ─────────────────────────────────────────────────────────────────────────────

# NVCC ?= $(shell which nvcc 2>/dev/null)
# ifeq ($(NVCC),)
#   NVCC := /usr/local/cuda/bin/nvcc
# endif

# OBJ_DIR     := obj
# IO_DIR      := I_O
# SOLVER_DIR  := solver
# MODELS_DIR  := models
# BIN         := rk45_solver

# DEBUG   ?= 0
# VERBOSE ?= 0

# HOST_INCLUDES   := -I. -I$(IO_DIR) -I$(SOLVER_DIR) -I$(MODELS_DIR)
# DEVICE_INCLUDES := -I. -I$(SOLVER_DIR) -I$(MODELS_DIR) -I$(IO_DIR)

# NVCCFLAGS := -std=c++17 -rdc=true \
#              -gencode arch=compute_80,code=sm_80 \
#              $(DEVICE_INCLUDES) -DUSE_MODEL_204 -Wno-deprecated-gpu-targets

# HOSTFLAGS := -std=c++17 $(HOST_INCLUDES) -DUSE_MODEL_204

# ifeq ($(DEBUG),1)
#   NVCCFLAGS += -g -G -O0
# else
#   NVCCFLAGS += -O2
# endif

# ifeq ($(VERBOSE),1)
#   Q :=
# else
#   Q := @
# endif

# # ─────────────────────────────────────────────────────────────────────────────
# # Source lists
# # ─────────────────────────────────────────────────────────────────────────────
# SRC_MAIN_CU  := main.cpp
# SRC_HOST_CPP := model_registry.cpp
# SRC_IO_CPP   := $(IO_DIR)/parameters_loader.cpp
# SRC_UTIL_CPP := $(MODELS_DIR)/ETmethods.cpp $(MODELS_DIR)/soiltemp.cpp
# SRC_CU       := $(SOLVER_DIR)/rk45_kernel.cu \
#                  $(SOLVER_DIR)/radau_kernel.cu \
#                  $(MODELS_DIR)/model_204_global.cu

# # ─────────────────────────────────────────────────────────────────────────────
# # Object files
# # ─────────────────────────────────────────────────────────────────────────────
# OBJ_MAIN     := $(OBJ_DIR)/main.o
# OBJ_HOST     := $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(SRC_HOST_CPP))
# OBJ_IO       := $(patsubst $(IO_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRC_IO_CPP))
# OBJ_UTIL     := $(patsubst $(MODELS_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRC_UTIL_CPP))
# OBJ_DEVICE   := $(patsubst %.cu,$(OBJ_DIR)/%.o,$(SRC_CU)) $(OBJ_MAIN)
# DEVICE_LINK  := $(OBJ_DIR)/device_link.o

# .PHONY: all clean

# all: $(BIN)
# 	@echo "Built $(BIN) [DEBUG=$(DEBUG) VERBOSE=$(VERBOSE)]"

# # ─────────────────────────────────────────────────────────────────────────────
# # Ensure obj/ and subdirs exist
# # ─────────────────────────────────────────────────────────────────────────────
# $(OBJ_DIR):
# 	@mkdir -p $(OBJ_DIR) $(OBJ_DIR)/$(SOLVER_DIR) $(OBJ_DIR)/$(MODELS_DIR)

# # ─────────────────────────────────────────────────────────────────────────────
# # 1) Compile main.cpp as CUDA (needed for <<<>>>)
# # ─────────────────────────────────────────────────────────────────────────────
# $(OBJ_MAIN): main.cpp | $(OBJ_DIR)
# 	$(Q)$(NVCC) $(NVCCFLAGS) -dc -x cu -c $< -o $@

# # ─────────────────────────────────────────────────────────────────────────────
# # 2) Compile I/O loader (parameters_loader.cpp) as plain host code
# # ─────────────────────────────────────────────────────────────────────────────
# $(OBJ_DIR)/%.o: $(IO_DIR)/%.cpp | $(OBJ_DIR)
# 	$(Q)$(NVCC) $(HOSTFLAGS) -c $< -o $@

# # ─────────────────────────────────────────────────────────────────────────────
# # 3) Compile other pure‐host .cpp files
# # ─────────────────────────────────────────────────────────────────────────────
# $(OBJ_DIR)/%.o: %.cpp | $(OBJ_DIR)
# 	$(Q)$(NVCC) $(HOSTFLAGS) -c $< -o $@

# # ─────────────────────────────────────────────────────────────────────────────
# # 4) Compile utility C++ that may need device relocatable code
# # ─────────────────────────────────────────────────────────────────────────────
# $(OBJ_DIR)/%.o: $(MODELS_DIR)/%.cpp | $(OBJ_DIR)
# 	$(Q)$(NVCC) $(HOSTFLAGS) -dc -c $< -o $@

# # ─────────────────────────────────────────────────────────────────────────────
# # 5) Compile .cu solver files → relocatable objects
# # ─────────────────────────────────────────────────────────────────────────────
# $(OBJ_DIR)/%.o: %.cu | $(OBJ_DIR)
# 	$(Q)$(NVCC) $(NVCCFLAGS) -dc $< -o $@

# # ─────────────────────────────────────────────────────────────────────────────
# # 6) Device‐link all relocatable objects
# # ─────────────────────────────────────────────────────────────────────────────
# $(DEVICE_LINK): $(OBJ_DEVICE)
# 	$(Q)$(NVCC) $(NVCCFLAGS) -dlink $^ -o $@

# # ─────────────────────────────────────────────────────────────────────────────
# # 7) Final link
# # ─────────────────────────────────────────────────────────────────────────────
# $(BIN): $(OBJ_HOST) $(OBJ_IO) $(OBJ_UTIL) $(OBJ_DEVICE) $(DEVICE_LINK)
# 	$(Q)$(NVCC) $(NVCCFLAGS) $^ -o $@

# # ─────────────────────────────────────────────────────────────────────────────
# # Clean
# # ─────────────────────────────────────────────────────────────────────────────
# clean:
# 	$(Q)rm -rf $(OBJ_DIR) $(BIN)
# 	@echo "Cleaned up object files and $(BIN)"

# ─────────────────────────────────────────────────────────────────────────────
# Makefile for rk45_solver (Model204 only)
# Disabled: ETmethods and soiltemp builds are commented out for now
# ─────────────────────────────────────────────────────────────────────────────

# Locate nvcc (allow override via environment)
NVCC ?= $(shell which nvcc 2>/dev/null)
ifeq ($(NVCC),)
  NVCC := /usr/local/cuda/bin/nvcc
endif

# Directories
OBJ_DIR     := obj
IO_DIR      := I_O
SOLVER_DIR  := solver
MODELS_DIR  := models
BIN         := rk45_solver

# Build modes
DEBUG   ?= 0
VERBOSE ?= 0

# Include paths
HOST_INCLUDES   := -I. -I$(IO_DIR) -I$(SOLVER_DIR) -I$(MODELS_DIR)
DEVICE_INCLUDES := -I. -I$(SOLVER_DIR) -I$(MODELS_DIR) -I$(IO_DIR)

# Compiler flags
# NVCCFLAGS := -std=c++17 -rdc=true \
#              -gencode arch=compute_80,code=sm_80 \
#              $(DEVICE_INCLUDES) -DUSE_MODEL_204 -Wno-deprecated-gpu-targets
# NVCCFLAGS := -std=c++17 -rdc=true \
#              -arch=sm_80 \
#              -gencode arch=compute_80,code=sm_80 \
#              $(DEVICE_INCLUDES) -DUSE_MODEL_204 -Wno-deprecated-gpu-targets


NVCCFLAGS := -std=c++17 -rdc=true \
-gencode arch=compute_80,code=sm_80 \
$(DEVICE_INCLUDES) -DUSE_MODEL_204 -Wno-deprecated-gpu-targets

# NVCCFLAGS := -std=c++17 -rdc=true -arch=sm_80 \
# $(DEVICE_INCLUDES) -DUSE_MODEL_204 -Wno-deprecated-gpu-targets


HOSTFLAGS := -std=c++17 $(HOST_INCLUDES) -DUSE_MODEL_204

ifeq ($(DEBUG),1)
  NVCCFLAGS += -g -G -O0
else
  NVCCFLAGS += -O2
endif

ifeq ($(VERBOSE),1)
  Q :=
else
  Q := @
endif

# ─────────────────────────────────────────────────────────────────────────────
# Source lists
# ─────────────────────────────────────────────────────────────────────────────
SRC_MAIN_CU  := main.cpp
SRC_HOST_CPP := model_registry.cpp
SRC_IO_CPP   := $(IO_DIR)/parameters_loader.cpp

# Temporarily disable ETmethods.cpp and soiltemp.cpp until host-device attributes are fixed
# SRC_UTIL_CPP := $(MODELS_DIR)/ETmethods.cpp $(MODELS_DIR)/soiltemp.cpp
SRC_UTIL_CPP :=

SRC_CU       := $(SOLVER_DIR)/rk45_kernel.cu \
                 $(SOLVER_DIR)/radau_kernel.cu \
                 $(MODELS_DIR)/model_204_global.cu

# ─────────────────────────────────────────────────────────────────────────────
# Object files
# ─────────────────────────────────────────────────────────────────────────────
OBJ_MAIN     := $(OBJ_DIR)/main.o
OBJ_HOST     := $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(SRC_HOST_CPP))
OBJ_IO       := $(patsubst $(IO_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRC_IO_CPP))
OBJ_UTIL     := $(patsubst $(MODELS_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRC_UTIL_CPP))
OBJ_DEVICE   := $(patsubst %.cu,$(OBJ_DIR)/%.o,$(SRC_CU)) $(OBJ_MAIN)
DEVICE_LINK  := $(OBJ_DIR)/device_link.o

.PHONY: all clean

all: $(BIN)
	@echo "Built $(BIN) [DEBUG=$(DEBUG) VERBOSE=$(VERBOSE)]"

# ─────────────────────────────────────────────────────────────────────────────
# Create object directories
# ─────────────────────────────────────────────────────────────────────────────
$(OBJ_DIR):
	@mkdir -p $(OBJ_DIR) $(OBJ_DIR)/$(SOLVER_DIR) $(OBJ_DIR)/$(MODELS_DIR)

# ─────────────────────────────────────────────────────────────────────────────
# 1) Compile main.cpp as CUDA (for <<<>>> kernels)
# ─────────────────────────────────────────────────────────────────────────────
$(OBJ_MAIN): main.cpp | $(OBJ_DIR)
	$(Q)$(NVCC) $(NVCCFLAGS) -dc -x cu -c $< -o $@

# ─────────────────────────────────────────────────────────────────────────────
# 2) Compile I/O loader as host-only
# ─────────────────────────────────────────────────────────────────────────────
$(OBJ_DIR)/%.o: $(IO_DIR)/%.cpp | $(OBJ_DIR)
	$(Q)$(NVCC) $(HOSTFLAGS) -c $< -o $@

# ─────────────────────────────────────────────────────────────────────────────
# 3) Compile other host-only .cpp files
# ─────────────────────────────────────────────────────────────────────────────
$(OBJ_DIR)/%.o: %.cpp | $(OBJ_DIR)
	$(Q)$(NVCC) $(HOSTFLAGS) -c $< -o $@

# ─────────────────────────────────────────────────────────────────────────────
# 4) Compile ETmethods.cpp & soiltemp.cpp (disabled)
# /*
# $(OBJ_DIR)/ETmethods.o: $(MODELS_DIR)/ETmethods.cpp | $(OBJ_DIR)
# 	$(Q)$(NVCC) $(HOSTFLAGS) -dc -c $< -o $@

# $(OBJ_DIR)/soiltemp.o: $(MODELS_DIR)/soiltemp.cpp | $(OBJ_DIR)
# 	$(Q)$(NVCC) $(HOSTFLAGS) -dc -c $< -o $@
# */
# ─────────────────────────────────────────────────────────────────────────────
# 5) Compile .cu solver files to relocatable objects
# ─────────────────────────────────────────────────────────────────────────────
$(OBJ_DIR)/%.o: %.cu | $(OBJ_DIR)
	$(Q)$(NVCC) $(NVCCFLAGS) -dc $< -o $@

# ─────────────────────────────────────────────────────────────────────────────
# 6) Device-link step
# ─────────────────────────────────────────────────────────────────────────────
$(DEVICE_LINK): $(OBJ_DEVICE)
	$(Q)$(NVCC) $(NVCCFLAGS) -dlink $^ -o $@

# ─────────────────────────────────────────────────────────────────────────────
# 7) Final link (omit OBJ_UTIL since ETmethods/soiltemp are disabled)
# ─────────────────────────────────────────────────────────────────────────────
$(BIN): $(OBJ_HOST) $(OBJ_IO) $(OBJ_DEVICE) $(DEVICE_LINK)
	$(Q)$(NVCC) $(NVCCFLAGS) $^ -o $@

# ─────────────────────────────────────────────────────────────────────────────
# Clean up
# ─────────────────────────────────────────────────────────────────────────────
clean:
	$(Q)rm -rf $(OBJ_DIR) $(BIN)
