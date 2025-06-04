# HLM-GPU

## Directory Structure

```text
root/
├── data/                       
│   ├── forcings/               # Forcing data (CSV/NetCDF)
│   │   ├── precip_forcing.nc   
│   │   ├── precip_lookup.csv
│   │   ├── t2m_forcing.nc
│   │   └── t2m_lookup.csv
│   ├── parameters.csv          # Spatially varying input parameters
│   └── config.yaml             # config for input/output paths, model ID,
│
├── scripts/                    # (Optional) post‐processing or plotting scripts
│   └── …                       
│
└── src/
    ├── Makefile                # Builds everything with nvcc (see below)
    │
    ├── main.cpp                # Host driver: calls setModelParameters<T>() and run_rk45<T>()
    ├── model_registry.hpp      # Declares inline setModelParameters<T>(…) and extern devParams
    ├── model_registry.cpp      # Defines the single __constant__ devParams for DummyModel
    │
    ├── I_O/                    # I/O utilities (e.g., CSV/NetCDF readers, checking input files)
    │   ├── forcing.cpp         # Reads forcing data (e.g., precipitation, temperature) from NetCDF files.
    │   └── forcing.hpp         # Declares functions for reading NetCDF forcing data.
    |   ├── parameters.cpp      # Reads spatially varying model parameters from CSV files.
    │   └── parameters.hpp      # Declares functions for reading parameter data from CSV files.
    |   ├── config_yaml.cpp     # Parses YAML configuration files for model settings and I/O paths.
    │   └── config_yaml.hpp     # Declares functions for parsing YAML configuration files.
    │
    ├── solver/                 # Core RK45 solver components
    │   ├── rk45.h              # Low‐level kernel prototype (templated kernel)
    │   ├── rk45_kernel.cu      # Implements rk45_kernel_multi<Model> for each Model
    │   ├── rk45_step_dense.cuh # Device‐side Dormand–Prince step (calls Model::rhs inside)
    │   └── rk45_api.hpp        # Host‐side “run_rk45<T>” and setModelParameters<T>() wrapper
    │
    └── models/                 
        ├── model_dummy.hpp     # Declares DummyModel::UID, Parameters, __device__ rhs(...), extern devParams
        └── model_dummy.cu      # Defines “__constant__ DummyModel::Parameters devParams;”
        └── … (future models go here, e.g. model_foo.hpp + model_foo.cu) …
```

## Building & Running

1. **Prerequisites**

* CUDA Toolkit (nvcc)
* C++14-capable compiler (e.g. `g++`)
* NetCDF C++ library for `.nc` files

2. **Compile**

```
cd src
make
```

This produces the executable `rk45_solver`.

### **Optional Build Modes**

You can customize the build process using the following options:

* **Debug Mode** (`DEBUG=1`): Enables debugging symbols (`-g`) and disables optimizations (`-O0`) for easier debugging.
* **Release Mode** (`DEBUG=0`): Enables optimizations (`-O2` or higher) for better performance.
* **Verbose Mode** (`VERBOSE=1`): Prints detailed compilation commands during the build process.

3. **Run**

```
./rk45_solver
```

By default, it uses `DummyModel::UID`, reads `data/forcing.csv` (if present), integrates `num_systems` ODEs from `t0=0.0` to `tf=5.0`, and writes:

* `final.csv` (one line per system: H0,H1,H2,H3,H4 at t=tf)
* `dense.csv` (“time,H0\_sys0,H1\_sys0,…,H4\_sysN” at `num_queries` sample times).

4. **Add a New Model**

* Create `models/model_new.hpp` following the stub in `model_dummy.hpp`.
* Add its `hostParams` block in `model_registry.cpp`.
* Add a `launch_rk45_new<<<…>>>` wrapper in `rk45_kernel.cu`.
* Rebuild with `make`.

5. **Adjust Tolerances or Coefficients**
   In `main.cpp`, before you call the kernel, set up

```
NewModel::Parameters hostParams = { absTol_val, relTol_val, /* ... */ };
cudaMemcpyToSymbol(NewModel::devParams, &hostParams, sizeof(hostParams));
```

so the GPU uses those tolerances instead of hard-coded values.
