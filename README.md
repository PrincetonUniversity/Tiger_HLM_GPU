# Tiger-HLM Runoff (GPU)
![Repo Status](https://img.shields.io/badge/status-active-brightgreen)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Tiger HLM Runoff** is a high-performance hydrologic model for simulating runoff at the hillslope level using GPU acceleration and distributed parallelism. It serves as the runoff generation component of the Tiger Hillslope Link Model (Tiger-HLM) framework. The complementary streamflow routing module is available at [Tiger HLM Routing](https://github.com/PrincetonUniversity/Tiger_HLM_Routing). This software can be used independently as well.

Please refer to the [wiki page](https://github.com/PrincetonUniversity/Tiger_HLM_GPU/wiki) for setup instructions and detailed documentation.


## **Build & Run**

### **1. Prerequisites**

Modules required: **CUDA Toolkit**, **C++14 compiler**, **NetCDF C++**, **MPI** (optional), and **NCO** (for post-processing).

**Setting Up the Environment**

Load the required CUDA toolkit using the following command:

For della gpu or gh use these as standard
```
module load cudatoolkit/12.9
module load openmpi/gcc/4.1.6
module load hdf5/gcc/openmpi-4.1.6/1.14.4
module load netcdf/gcc/hdf5-1.14.4/openmpi-4.1.6/4.9.2

```


### **2. Build**

Clean and compile:

```bash
make clean
make
```

Optional flags:

* `DEBUG=1` → Debug build
* `DEBUG=0` → Release build (default)
* `VERBOSE=1` → Show build commands



### **3. Run**

Serial:

```bash
./bin/runoff /path/to/config.yaml
```

MPI:

```bash
mpirun -np 2 ./bin/runoff /path/to/config.yaml
```


## Directory Structure

```text
root/
├── data/                        # Input data folder             
│   ├──Stony_Brook/              # Example for Stony Brook basin
│   ├── precipitation_forcing.nc # Forcing data (NETCDF)
│   ├── precipitation_lookup.csv # Forcing lookup (CSV)
│   ├── temperature_forcing.nc   # Forcing data (NETCDF)
│   ├── temperature_lookup.csv	# Forcing 2 lookup (CSV)
│   ├── parameters.csv          # Spatially varying input parameters (CSV)
│   └── config.yaml             # configuration file for input/output paths, model ID, 
│
├── scripts/                    # pre-processing scripts
│   └── generate_lookup_tables.ipynb # code to generate lookup tables from forcings NETCDF files and params.csv                 
│
└── src/
    ├── Makefile                # Builds everything with nvcc (see below)
    │
    ├── main.cpp                # Host driver: calls setModelParameters<T>() and run_rk45<T>()
    ├── simulation_driver.hpp                # Host driver: calls setModelParameters<T>() and run_rk45<T>()
    ├── stream.hpp
    │
    ├── I_O/                    # I/O utilities (e.g., CSV/NetCDF readers, checking input files)
    |   ├── config_loader.cpp     # Parses YAML configuration files for model settings and I/O paths
    │   └── config_loader.hpp     # Declares functions for parsing YAML configuration files
    │   ├── forcing_data.cu
    │   ├── forcing_data.h        
    │   ├── forcing_loader.cpp         # Reads forcing data (e.g., precipitation, temperature) from NetCDF files
    │   └── forcing_loader.hpp         # Declares functions for reading NetCDF forcing data
    |   ├── parameters_loader.cpp      # Reads spatially varying model parameters from CSV files
    │   ├── parameters_loader.hpp      # Declares functions for reading parameter data from CSV files
    │   ├── output_serie.cpp
    │   └── output_series.hpp
    │
    ├── solver/                 # Core RK45 solver components
    │   ├── rk45.h              # Low‐level kernel prototype (templated kernel)
    │   ├── rk45_kernel.cu      # Implements rk45_kernel_multi<Model> for each Model
    │   ├── rk45_step_dense.cuh # Device‐side Dormand–Prince step (calls Model::rhs inside) and dense output
    │   └── rk45_api.hpp        # Host‐side “run_rk45<T>” and setModelParameters<T>() wrapper
    │
    └──  models/                 
        ├── model_204_global.cu     # GPU constants for Runoff5 parameters
        ├── model_204.hpp
    	├── model_registry.hpp		# Model equations for Runoff
		├── model_registry.cpp		# Defines the single __constant__ devParams for DummyModel   
        ├── ETmethods.cpp  			# Evapotranspiration formulas (Hamon, ETactual)
        ├── ETmethods.hpp
        ├── soiltemp.cpp 			# Implements soil temperature change at daily scale
        └── soiltemp.hpp

```


# Citation
To cite this software in your publication, please use the following BibTeX (to be updated upon paper acceptance) to refer to the code's [method paper](empty):
```
@article{,
	doi = {},
	url = {},
	year = ,
	month = ,
	publisher = {},
	volume = {},
	number = {},
	pages = {}
	author = {},
	title = {},
	journal = {},
}
```

Finally, we have DOIs for each released version on Zenodo. This approach promotes computational reproducibility by allowing you to specify the exact version of the code used to generate the results presented in your publication. Please use the following citation for citing the software.

```bibtex
@software{binjolkar_2026_16696727,
  author       = {Binjolkar, Manjaree and
                  Michalek, Alexander and
                  Amorim, Renato and
                  Li, Donghui and
                  Maebius, Sarah and
                  Pu, Tianjiao and
                  Ethier, Stephane and
                  Villarini, Gabriele},
  title        = {Tiger-HLM: A GPU-Accelerated Distributed
                  Hydrologic Model},
  month        = aug,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.16696727},
  url          = {https://doi.org/10.5281/zenodo.16696727},
}
```
