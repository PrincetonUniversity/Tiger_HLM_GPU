# --------------------------------------------------------------------------------------------------
# Compiler flags (include solver/ and models/)
# --------------------------------------------------------------------------------------------------
DEBUG ?= 0
VERBOSE ?= 0

NVCCFLAGS := -std=c++14 -arch=sm_70 -rdc=true -Wno-deprecated-gpu-targets -I. -I./solver -I./models
ifeq ($(DEBUG), 1)
	NVCCFLAGS += -g -G
else
	NVCCFLAGS += -O2
endif

# Ensure critical include paths are present
ifeq (,$(findstring -I./models,$(NVCCFLAGS)))
$(error "Include path for models/ is missing in NVCCFLAGS")
endif

ifeq (,$(findstring -I./solver,$(NVCCFLAGS)))
$(error "Include path for solver/ is missing in NVCCFLAGS")
endif

# Verbose mode
Q := @
ifeq ($(VERBOSE), 1)
Q :=
endif

# --------------------------------------------------------------------------------------------------
# Compiler
# --------------------------------------------------------------------------------------------------
NVCC := nvcc

# --------------------------------------------------------------------------------------------------
# Directories & Targets
# --------------------------------------------------------------------------------------------------
OBJ_DIR    := obj
BIN        := rk45_solver

SOLVER_DIR := solver
MODELS_DIR := models

# --------------------------------------------------------------------------------------------------
# Source files
# --------------------------------------------------------------------------------------------------
SRCS_HOST := \
	main.cpp \
	model_registry.cpp

SRCS_DEVICE := \
	$(SOLVER_DIR)/rk45_kernel.cu \
	$(MODELS_DIR)/model_dummy_global.cu

# --------------------------------------------------------------------------------------------------
# Object filenames
# --------------------------------------------------------------------------------------------------
OBJS_HOST := \
	$(OBJ_DIR)/main.o \
	$(OBJ_DIR)/model_registry.o

OBJS_DEVICE := \
	$(OBJ_DIR)/rk45_kernel.o \
	$(OBJ_DIR)/model_dummy.o

# --------------------------------------------------------------------------------------------------
# Build rules
# --------------------------------------------------------------------------------------------------
all: $(BIN)
	@echo "Build complete: $(BIN)"

# Ensure obj/ directory exists
$(OBJ_DIR):
	@mkdir -p $(OBJ_DIR)

# 1) Compile main.cpp with nvcc (it launches kernels via <<<>>>)
$(OBJ_DIR)/main.o: main.cpp model_registry.hpp \
	               $(SOLVER_DIR)/rk45.h $(SOLVER_DIR)/rk45_step_dense.cuh $(SOLVER_DIR)/rk45_api.hpp \
	               $(MODELS_DIR)/model_dummy.hpp | $(OBJ_DIR)
	$(Q)$(NVCC) $(NVCCFLAGS) -x cu -c main.cpp -o $@

# 2) Compile model_registry.cpp (contains cudaMemcpyToSymbol(devParams,…))
$(OBJ_DIR)/model_registry.o: model_registry.cpp model_registry.hpp $(MODELS_DIR)/model_dummy.hpp | $(OBJ_DIR)
	$(Q)$(NVCC) $(NVCCFLAGS) -x cu -c model_registry.cpp -o $@

# 3) Compile rk45_kernel.cu (templated device code)
$(OBJ_DIR)/rk45_kernel.o: $(SOLVER_DIR)/rk45_kernel.cu \
	                     $(SOLVER_DIR)/rk45.h $(SOLVER_DIR)/rk45_step_dense.cuh $(SOLVER_DIR)/rk45_api.hpp \
	                     $(MODELS_DIR)/model_dummy.hpp | $(OBJ_DIR)
	$(Q)$(NVCC) $(NVCCFLAGS) -dc $(SOLVER_DIR)/rk45_kernel.cu -o $@

# 4) Compile model_dummy_global.cu (the one definition of devParams)
$(OBJ_DIR)/model_dummy.o: $(MODELS_DIR)/model_dummy_global.cu $(MODELS_DIR)/model_dummy.hpp | $(OBJ_DIR)
	$(Q)$(NVCC) $(NVCCFLAGS) -x cu -c $(MODELS_DIR)/model_dummy_global.cu -o $@

# 5) Link everything
$(BIN): $(OBJS_HOST) $(OBJS_DEVICE)
	$(Q)$(NVCC) $(NVCCFLAGS) $(OBJS_DEVICE) $(OBJS_HOST) -o $(BIN)

# 6) Clean up
clean:
	$(Q)rm -rf $(OBJ_DIR)/*.o $(BIN)

distclean: clean
	$(Q)rm -rf $(OBJ_DIR)/*.d

# Check for CUDA Toolkit installation
ifeq ($(shell which nvcc),)
$(error "CUDA Toolkit (nvcc) not found. Please install it or add it to your PATH. Visit https://developer.nvidia.com/cuda-toolkit for installation instructions.")
endif

.PHONY: all clean distclean