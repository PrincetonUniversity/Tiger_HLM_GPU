//src/stream.hpp
#pragma once

#include "parameters_loader.hpp"
#include <array>

/**
 * @brief A Stream represents one spatial “hillslope” (or channel) for Model204.
 *
 * Each Stream holds:
 *   - an integer @c id identifying this stream,
 *   - an integer @c next_id pointing to the downstream neighbor,
 *   - a @c SpatialParams struct @c sp containing all site-specific parameters
 *     (e.g. conductivity, slope, PET coefficients, etc.), and
 *   - an initial condition vector @c y0 of length Model::N_EQ (h_snow, h_static,
 *     h_surface, h_grav, h_aquifer).
 *
 *
 * @tparam Model  The ODE model struct (e.g. Model204) defining N_EQ.
 */
template<class Model>
struct Stream {
    long id;                     // Unique identifier for this stream
    long next_id;                // ID of downstream (next) stream 
    
    SpatialParams sp;            // Per-stream spatial parameters
                                 // fields match Model204::rhs’s stubbed inputs)
    
    std::array<double, Model::N_EQ> y0;  // Initial state [h_snow, h_static, h_surface, h_grav, h_aquifer]

    /**
     * @brief Construct a Stream from a SpatialParams record and a common initial state.
     *
     * @param _sp   The spatial parameters row loaded from CSV.
     * @param _y0   A std::array<double,N_EQ> providing the initial y values.
     */
    Stream(SpatialParams const& _sp,
           std::array<double,Model::N_EQ> const& _y0)
      : id(_sp.stream)
      , next_id(_sp.next_stream)
      , sp(_sp)
      , y0(_y0)
    {}
};
