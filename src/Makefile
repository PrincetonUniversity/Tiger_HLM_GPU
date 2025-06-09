# --------------------------------------------------------------------------------------------------
# Compiler & directories
# --------------------------------------------------------------------------------------------------
NVCC      := nvcc
OBJ_DIR   := obj
BIN       := rk45_solver

SOLVER_DIR := solver
MODELS_DIR := models
IO_DIR     := I_O

# --------------------------------------------------------------------------------------------------
# Choose which model to build: "dummy" or "204"
# --------------------------------------------------------------------------------------------------
MODEL ?= 204

# --------------------------------------------------------------------------------------------------
# Macros for both host and device compiles
# --------------------------------------------------------------------------------------------------
# Ensure the host parser (Clang/C++ front-end) also gets the same USE_MODEL_* define:
HOST_CPPFLAGS := -DUSE_MODEL_$(MODEL) -I./$(IO_DIR)

# --------------------------------------------------------------------------------------------------
# Compiler flags (include solver/, models/, I_O/) and the model selector macro
# --------------------------------------------------------------------------------------------------
DEBUG   ?= 0
VERBOSE ?= 0

NVCCFLAGS := -std=c++17 -arch=sm_70 -rdc=true -Wno-deprecated-gpu-targets \
             -I. -I./$(SOLVER_DIR) -I./$(MODELS_DIR) -I./$(IO_DIR)

ifeq ($(DEBUG),1)
	NVCCFLAGS += -g -G
else
	NVCCFLAGS += -O2
endif

# Pull in the right model switch for active_model.hpp
ifeq ($(MODEL),dummy)
	NVCCFLAGS += -DUSE_MODEL_DUMMY
else ifeq ($(MODEL),204)
	NVCCFLAGS += -DUSE_MODEL_204
else
	$(error Unknown MODEL '$(MODEL)'; valid: dummy or 204)
endif

# Ensure include paths
ifeq (,$(findstring -I./$(MODELS_DIR),$(NVCCFLAGS)))
	$(error "Include path for $(MODELS_DIR)/ is missing in NVCCFLAGS")
endif
ifeq (,$(findstring -I./$(SOLVER_DIR),$(NVCCFLAGS)))
	$(error "Include path for $(SOLVER_DIR)/ is missing in NVCCFLAGS")
endif
ifeq (,$(findstring -I./$(IO_DIR),$(NVCCFLAGS)))
	$(error "Include path for $(IO_DIR)/ is missing in NVCCFLAGS")
endif

# Verbose?
Q := @
ifeq ($(VERBOSE),1)
	Q :=
endif

# --------------------------------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------------------------------
# Common host sources
SRCS_HOST := \
	main.cpp \
	model_registry.cpp \
	$(IO_DIR)/parameters_loader.cpp

# For Model204 only: ETmethods & soiltemp
ifeq ($(MODEL),204)
	SRCS_HOST += \
		$(MODELS_DIR)/ETmethods.cpp \
		$(MODELS_DIR)/soiltemp.cpp
endif

# Device sources
SRCS_DEVICE := \
	$(SOLVER_DIR)/rk45_kernel.cu \
	$(SOLVER_DIR)/radau_kernel.cu

# Pull in all per-model globals automatically:
MODEL_GLOBAL_SRCS := $(wildcard $(MODELS_DIR)/*_global.cu)
MODEL_GLOBAL_OBJS := $(patsubst $(MODELS_DIR)/%.cu,$(OBJ_DIR)/%.o,$(MODEL_GLOBAL_SRCS))

# --------------------------------------------------------------------------------------------------
# Objects
# --------------------------------------------------------------------------------------------------
# Host objects (flattened into obj/)
OBJS_HOST := \
	$(OBJ_DIR)/main.o \
	$(OBJ_DIR)/model_registry.o \
	$(OBJ_DIR)/parameters_loader.o

ifeq ($(MODEL),204)
	OBJS_HOST += \
		$(OBJ_DIR)/ETmethods.o \
		$(OBJ_DIR)/soiltemp.o
endif

# Device objects (all compiled with -dc)
OBJS_DEVICE := \
	$(OBJ_DIR)/rk45_kernel.o \
	$(OBJ_DIR)/radau_kernel.o \
	$(MODEL_GLOBAL_OBJS)

# Device-link object
DEVICE_LINK_OBJ := $(OBJ_DIR)/device_link.o

# --------------------------------------------------------------------------------------------------
# Build rules
# --------------------------------------------------------------------------------------------------
all: $(BIN)
	@echo "Built $(BIN) [MODEL=$(MODEL)]"

# ensure obj directory
$(OBJ_DIR):
	@mkdir -p $(OBJ_DIR)

# ---------------------
# Host compilation
# ---------------------
# .cpp → obj/%.o
$(OBJ_DIR)/%.o: %.cpp | $(OBJ_DIR)
	$(Q)$(NVCC) $(HOST_CPPFLAGS) $(NVCCFLAGS) -x cu -c $< -o $@

# I_O/parameters_loader.cpp → obj/parameters_loader.o
$(OBJ_DIR)/parameters_loader.o: $(IO_DIR)/parameters_loader.cpp $(IO_DIR)/parameters_loader.hpp | $(OBJ_DIR)
	$(Q)$(NVCC) $(HOST_CPPFLAGS) $(NVCCFLAGS) -x cu -c $< -o $@

# ETmethods.cpp → obj/ETmethods.o
$(OBJ_DIR)/ETmethods.o: $(MODELS_DIR)/ETmethods.cpp $(MODELS_DIR)/ETmethods.hpp | $(OBJ_DIR)
	$(Q)$(NVCC) $(HOST_CPPFLAGS) $(NVCCFLAGS) -dc -x cu -c $< -o $@

# soiltemp.cpp → obj/soiltemp.o
$(OBJ_DIR)/soiltemp.o: $(MODELS_DIR)/soiltemp.cpp $(MODELS_DIR)/soiltemp.hpp | $(OBJ_DIR)
	$(Q)$(NVCC) $(HOST_CPPFLAGS) $(NVCCFLAGS) -dc -x cu -c $< -o $@

# ---------------------
# Device compilation
# ---------------------
# rk45_kernel.cu → obj/rk45_kernel.o
$(OBJ_DIR)/rk45_kernel.o: $(SOLVER_DIR)/rk45_kernel.cu \
                          $(SOLVER_DIR)/rk45.h \
                          $(SOLVER_DIR)/rk45_step_dense.cuh \
                          $(SOLVER_DIR)/event_detector.cuh \
                          $(SOLVER_DIR)/small_lu.cuh \
                          $(SOLVER_DIR)/radau_step_dense.cuh \
                          $(SOLVER_DIR)/radau_kernel.cuh \
                          $(SOLVER_DIR)/rk45_api.hpp \
                          $(MODELS_DIR)/active_model.hpp | $(OBJ_DIR)
	$(Q)$(NVCC) $(NVCCFLAGS) -dc $< -o $@

# radau_kernel.cu → obj/radau_kernel.o
$(OBJ_DIR)/radau_kernel.o: $(SOLVER_DIR)/radau_kernel.cu \
                           $(SOLVER_DIR)/radau_kernel.cuh \
                           $(SOLVER_DIR)/radau_step_dense.cuh \
                           $(SOLVER_DIR)/small_lu.cuh \
                           $(SOLVER_DIR)/event_detector.cuh \
                           $(MODELS_DIR)/active_model.hpp | $(OBJ_DIR)
	$(Q)$(NVCC) $(NVCCFLAGS) -dc $< -o $@

# model_*_global.cu → obj/model_*_global.o (auto via wildcard)
$(OBJ_DIR)/%_global.o: $(MODELS_DIR)/%_global.cu $(MODELS_DIR)/active_model.hpp | $(OBJ_DIR)
	$(Q)$(NVCC) $(NVCCFLAGS) -dc -x cu -c $< -o $@

# ---------------------
# Device-link and final link
# ---------------------
# Stitch all -dc objects into one fatbinary
$(DEVICE_LINK_OBJ): $(OBJS_DEVICE)
	$(Q)$(NVCC) $(NVCCFLAGS) -dlink $^ -o $@

# Final link: include device_link.o
$(BIN): $(OBJS_HOST) $(OBJS_DEVICE) $(DEVICE_LINK_OBJ)
	$(Q)$(NVCC) $(NVCCFLAGS) $^ -o $@

clean:
	$(Q)rm -rf $(OBJ_DIR)/*.o $(BIN)

distclean: clean
	$(Q)rm -rf $(OBJ_DIR)/*.d

# Sanity check: nvcc must exist
ifeq ($(shell which nvcc),)
	$(error "nvcc not found. Please install CUDA Toolkit or add it to PATH.")
endif

.PHONY: all clean distclean
