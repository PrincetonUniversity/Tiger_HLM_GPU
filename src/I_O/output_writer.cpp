// /I_O/output_writer.cpp
#include "output_writer.hpp"
#include "models/model_204.hpp"  // for STATE_SURF_RUNOFF, STATE_TOTAL_RUNOFF

StreamingNetCDF::StreamingNetCDF(std::string const&            filename,
                                 std::vector<uint32_t> const&  linkIDs,
                                 std::vector<int>      const&  stateIDs,
                                 int                            full_N_EQ,
                                 int                            mode,
                                 std::vector<SelectedVar> const& selVars,
                                 int                            deflate)
  : fn_(filename)
  , mode_(mode)
  , sysN_(int(linkIDs.size()))
  , varN_(full_N_EQ)
  , sysIDs_(linkIDs)
  , varIDs_(stateIDs)
  , selVars_(selVars)
{
  openCreate(deflate);
}

StreamingNetCDF::~StreamingNetCDF() {
  if (ncid_ >= 0) {
    nc_close(ncid_);
  }
}

void StreamingNetCDF::openCreate(int deflate) {
  // Create & define dims
  NC_CHECK(nc_create(fn_.c_str(), NC_NETCDF4|NC_CLOBBER, &ncid_));
  NC_CHECK(nc_def_dim(ncid_, "system",   sysN_, &sysDim_));
  NC_CHECK(nc_def_dim(ncid_, "time",     UNLIM, &timeDim_));
  if (mode_ & DENSE) {
    NC_CHECK(nc_def_dim(ncid_, "variable", varN_, &varDim_));
  }

  // Define coordinate vars
  NC_CHECK(nc_def_var(ncid_, "system",   NC_UINT,   1, &sysDim_,  &sysVar_));
  NC_CHECK(nc_def_var(ncid_, "time",     NC_DOUBLE, 1, &timeDim_, &timeVar_));
  if (mode_ & DENSE) {
    NC_CHECK(nc_def_var(ncid_, "variable", NC_INT,    1, &varDim_,  &varVar_));
  }

  // Define data variables
  if (mode_ & DENSE) {
    int d3[3] = {sysDim_, timeDim_, varDim_};
    NC_CHECK(nc_def_var(ncid_, "outputs", NC_DOUBLE, 3, d3, &denseVar_));
    if (deflate) NC_CHECK(nc_def_var_deflate(ncid_, denseVar_, 1,1,deflate));
  }
  if (mode_ & RUNOFF) {
    int d2[2] = {sysDim_, timeDim_};
    NC_CHECK(nc_def_var(ncid_, "surface_runoff", NC_DOUBLE, 2, d2, &surfVar_));
    NC_CHECK(nc_def_var(ncid_, "total_runoff",   NC_DOUBLE, 2, d2, &totalVar_));
    if (deflate) {
      NC_CHECK(nc_def_var_deflate(ncid_, surfVar_,  1,1,deflate));
      NC_CHECK(nc_def_var_deflate(ncid_, totalVar_, 1,1,deflate));
    }
  }
  if (mode_ & SELECTED) {
    int d2[2] = {sysDim_, timeDim_};
    selVarIds_.resize(selVars_.size());
    for (size_t i = 0; i < selVars_.size(); ++i) {
      NC_CHECK(nc_def_var(ncid_,
                          selVars_[i].name.c_str(),
                          NC_DOUBLE,
                          2, d2,
                          &selVarIds_[i]));
      if (deflate)
        NC_CHECK(nc_def_var_deflate(ncid_, selVarIds_[i], 1,1,deflate));
    }
  }

  // Leave define mode
  NC_CHECK(nc_enddef(ncid_));

  // Write the always‐static coords
  NC_CHECK(nc_put_var_uint(ncid_, sysVar_, sysIDs_.data()));
  if (mode_ & DENSE) {
    NC_CHECK(nc_put_var_int(ncid_, varVar_, varIDs_.data()));
  }
}

void StreamingNetCDF::appendSlab(const double* h_dense,
                                 const double* timeBuf,
                                 int            t0,
                                 int            tN)
{
  // Write time slab
  size_t start1[1] = {size_t(t0)}, count1[1] = {size_t(tN)};
  NC_CHECK(nc_put_vara_double(ncid_, timeVar_, start1, count1, timeBuf));

  // Dense 3D:
  if (mode_ & DENSE) {
    size_t start3[3] = {0, size_t(t0), 0};
    size_t count3[3] = {size_t(sysN_), size_t(tN), size_t(varN_)};
    NC_CHECK(nc_put_vara_double(ncid_, denseVar_, start3, count3, h_dense));
  }

  // Runoff 2D:
  if (mode_ & RUNOFF) {
    std::vector<double> surf(sysN_*tN), tot(sysN_*tN);
    for (int s = 0; s < sysN_; ++s) {
      for (int tt = 0; tt < tN; ++tt) {
        size_t base = (size_t(s)*tN + tt)*varN_;
        surf[size_t(s)*tN + tt] = h_dense[base + Model204::STATE_SURF_RUNOFF];
        tot [size_t(s)*tN + tt] = h_dense[base + Model204::STATE_TOTAL_RUNOFF];
      }
    }
    size_t start2[2] = {0, size_t(t0)}, count2[2] = {size_t(sysN_), size_t(tN)};
    NC_CHECK(nc_put_vara_double(ncid_, surfVar_,  start2, count2, surf.data()));
    NC_CHECK(nc_put_vara_double(ncid_, totalVar_, start2, count2, tot.data()));
  }

  // Selected slices:
  if (mode_ & SELECTED) {
    size_t start2[2] = {0, size_t(t0)}, count2[2] = {size_t(sysN_), size_t(tN)};
    for (size_t i = 0; i < selVars_.size(); ++i) {
      const auto& sv = selVars_[i];
      std::vector<double> buf(sysN_*tN);
      for (int s = 0; s < sysN_; ++s) {
        for (int tt = 0; tt < tN; ++tt) {
          size_t base = (size_t(s)*tN + tt)*varN_;
          buf[size_t(s)*tN + tt] = h_dense[base + sv.idx];
        }
      }
      NC_CHECK(nc_put_vara_double(ncid_, selVarIds_[i], start2, count2, buf.data()));
    }
  }
}
