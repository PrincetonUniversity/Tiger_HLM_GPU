// I_O/config_loader.hpp
// Trial code for config file
#pragma once

#include <string>
#include <chrono>
#include <vector>

struct SimulationConfig {
    struct ModelInfo {
        int   uid;
        std::string name;
    } model;

    struct TimeInfo {
        std::chrono::system_clock::time_point start;
        std::chrono::system_clock::time_point end;
    } time;

    struct InitialInfo {
        std::string mode;   // "cold" or "hot"
        std::string file;   // only if mode=="hot"
    } initial;

    struct ForcingInfo {
        std::string type;      // e.g. "folder_nc"
        std::string path;
        std::string lookup_csv;
        std::string var_precip;
        std::string var_temp;
    } forcings;

    struct OutputInfo {
        std::string print_interval;
        std::vector<int> states;
    } output;

    struct SolverInfo {
        double rtol, atol, safety, min_scale, max_scale;
        bool   override_tolerances;
        double initial_step;
        bool   override_initial_step;
    } solver;

    struct MPIInfo {
        int step_storage;
        int transfer_buffer;
        int discontinuity_buf;
    } mpi;

    struct FlagsInfo {
        bool uses_dam;
        bool convert_area;
    } flags;
};

SimulationConfig load_config(const std::string& filename);
