// I_O/config_loader.cpp
// Trial code for config file
#include "include/config_loader.hpp"
#include <yaml-cpp/yaml.h>
#include <stdexcept>
#include <sstream>
#include <iomanip>

using namespace std::chrono;

static system_clock::time_point parse_iso8601(const std::string& s) {
    std::tm tm = {};
    std::istringstream ss(s);
    ss >> std::get_time(&tm, "%Y-%m-%dT%H:%M:%S");
    if (ss.fail()) throw std::runtime_error("Failed to parse time: " + s);
    return system_clock::from_time_t(std::mktime(&tm));
}

SimulationConfig load_config(const std::string& filename) {
    YAML::Node doc = YAML::LoadFile(filename);
    SimulationConfig cfg;

    // 1) model
    auto m = doc["model"];
    cfg.model.uid  = m["uid"].as<int>();
    cfg.model.name = m["name"].as<std::string>("");

    // 2) time
    auto t = doc["time"];
    cfg.time.start = parse_iso8601(t["start"].as<std::string>());
    cfg.time.end   = parse_iso8601(t["end"  ].as<std::string>());

    // 3) initial
    auto init = doc["initial"];
    cfg.initial.mode = init["mode"].as<std::string>();
    if (cfg.initial.mode == "hot")
        cfg.initial.file = init["file"].as<std::string>();

    // 6) forcings
    auto f = doc["forcings"];
    cfg.forcings.type       = f["type"].as<std::string>();
    cfg.forcings.path       = f["path"].as<std::string>();
    cfg.forcings.lookup_csv = f["lookup"].as<std::string>();
    auto vars = f["vars"];
    cfg.forcings.var_precip = vars["precipitation"].as<std::string>();
    cfg.forcings.var_temp   = vars["temperature"].as<std::string>();

    // 7) output
    auto o = doc["output"];
    cfg.output.print_interval = o["print_interval"].as<std::string>();
    if (o["states"]) {
        for (auto v : o["states"])
            cfg.output.states.push_back(v.as<int>());
    }

    // 8) solver
    auto s = doc["solver"];
    YAML::Node tol = s["tolerances"];
    if (tol) {
        cfg.solver.override_tolerances = true;
        cfg.solver.rtol      = tol["rtol"     ].as<double>();
        cfg.solver.atol      = tol["atol"     ].as<double>();
        cfg.solver.safety    = tol["safety"   ].as<double>();
        cfg.solver.min_scale = tol["min_scale"].as<double>();
        cfg.solver.max_scale = tol["max_scale"].as<double>();
    }
    if (s["initial_step"] && !s["initial_step"].IsNull()) {
        cfg.solver.override_initial_step = true;
        cfg.solver.initial_step = s["initial_step"].as<double>();
    }

    // 9) mpi // need to change this to parallel commands on GPU/CPu 
    auto mpi = doc["mpi"];
    cfg.mpi.step_storage     = mpi["step_storage"].as<int>();
    cfg.mpi.transfer_buffer  = mpi["transfer_buffer"].as<int>();
    cfg.mpi.discontinuity_buf= mpi["discontinuity_buf"].as<int>();

    // 10) flags // not used yet, but can be added later
    // auto fl = doc["flags"];
    // cfg.flags.uses_dam     = fl["uses_dam"].as<bool>();
    // cfg.flags.convert_area = fl["convert_area"].as<bool>();

    return cfg;
}
