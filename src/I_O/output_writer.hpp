// /I_O/output_writer.hpp
#pragma once

#include <netcdf.h>
#include <string>
#include <vector>
#include <cstdint>
#include <stdexcept>

/** Macro to check NetCDF calls and throw on error. */
#define NC_CHECK(call)                                     \
  do {                                                     \
    int _st = (call);                                      \
    if (_st != NC_NOERR)                                   \
      throw std::runtime_error(nc_strerror(_st));         \
  } while (0)

static constexpr size_t UNLIM = NC_UNLIMITED;

/** Bitflags for which outputs to enable. */
enum WriteMode {
  DENSE    = 1<<0,  ///< full 3D [system×time×variable]
  RUNOFF   = 1<<1,  ///< 2D surface + total runoff
  SELECTED = 1<<2   ///< arbitrary selected 2D vars
};

/** One selected‐state → NetCDF variable name. */
struct SelectedVar {
  int         idx;   ///< index in the last dimension of the dense array
  std::string name;  ///< netCDF variable name
};

/**
 * @brief Streaming NetCDF writer for Model204 outputs.
 *
 * Creates a single .nc with unlimited time, and lets you append
 * one “slab” at a time without pre‐knowing the total length.
 */
class StreamingNetCDF {
public:
  /**
   * @brief Create (or overwrite) the output file and define dims/vars.
   *
   * @param filename   Path to the .nc to create.
   * @param linkIDs    Vector of length num_systems with your stream IDs.
   * @param stateIDs   Vector of length full_N_EQ with state‐var IDs.
   * @param full_N_EQ  Total number of state variables in your model.
   * @param mode       Bitmask of WriteMode flags (DENSE|RUNOFF|SELECTED).
   * @param selVars    List of SelectedVar structs (used if SELECTED).
   * @param deflate    Zlib compression level (0=no, 1–9=zlib).
   *
   * @throws std::runtime_error on any NetCDF error.
   */
  StreamingNetCDF(std::string const&            filename,
                  std::vector<uint32_t> const&  linkIDs,
                  std::vector<int>      const&  stateIDs,
                  int                            full_N_EQ,
                  int                            mode,
                  std::vector<SelectedVar> const& selVars,
                  int                            deflate = 4);

  /**
   * @brief Close the .nc.  Automatically called in the destructor.
   */
  ~StreamingNetCDF();

  /**
   * @brief Append a time‐slab of output.
   *
   * @param h_dense   Pointer to double[ system × tN × full_N_EQ ]
   * @param timeBuf   Pointer to double[tN] of time values (minutes).
   * @param t0        Starting index in the unlimited time dimension.
   * @param tN        Number of time‐steps in this slab.
   *
   * @throws std::runtime_error on any NetCDF error.
   */
  void appendSlab(const double* h_dense,
                  const double* timeBuf,
                  int            t0,
                  int            tN);

private:
  void openCreate(int deflate);

  std::string               fn_;
  int                       ncid_     = -1;
  int                       mode_;
  int                       sysN_, varN_;
  int                       sysDim_, timeDim_, varDim_;
  int                       sysVar_, timeVar_, varVar_;
  int                       denseVar_, surfVar_, totalVar_;
  std::vector<int>          selVarIds_;
  std::vector<uint32_t>     sysIDs_;
  std::vector<int>          varIDs_;
  std::vector<SelectedVar>  selVars_;
};
