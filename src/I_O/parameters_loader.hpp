//I_O/parameters_loader.hpp
#pragma once
#include <string>
#include <vector>

// -----------------------------------------------------------------------------
// SpatialParams
// -----------------------------------------------------------------------------
// Holds all of the per‐stream, spatially varying parameters that Model204::rhs
// expects.  The CSV must have these columns (case‐sensitive):
//
//   stream,next_stream,
//   c1,infil,perco,Hu,lat,sw,ss,
//   n_mann,slope,L,A_h,alpha3,alpha4,
//   melt_f,temp_thr
//
// where `stream` and `next_stream` are integer IDs, and the rest are doubles.
// -----------------------------------------------------------------------------
struct SpatialParams {
    long   stream;
    long   next_stream;
    double c1;
    double infil;
    double perco;
    double Hu;
    double lat;
    double sw;
    double ss;
    double n_mann;
    double slope;
    double L;
    double A_h;
    double alpha3;
    double alpha4;
    double melt_f;
    double temp_thr;
};


// -----------------------------------------------------------------------------
// loadSpatialParams
// -----------------------------------------------------------------------------
// Reads `csv_path` (must be UTF‐8, with comma separators, and a header row as above)
// and returns a vector of SpatialParams, one per data row.
// Throws std::runtime_error on I/O or parse errors.
// -----------------------------------------------------------------------------
std::vector<SpatialParams> loadSpatialParams(const std::string& csv_path);