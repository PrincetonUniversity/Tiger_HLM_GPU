#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <ctime>      
#include <cctype>     
#include <cstdio>     
#include <regex>
#include <filesystem>
#include <cstring>    
#include <cmath>      

#include "forcing_loader.hpp"

namespace fs = std::filesystem;

// ─────────── filename date range parser ───────────
struct DateRange { int y1,m1,d1, y2,m2,d2; };

static bool parse_date_range_from_name(const std::string& name, DateRange& out) {
    // finds ...YYYYMMDD_YYYYMMDD... anywhere in filename
    static const std::regex re(R"((\d{8})_(\d{8}))");
    std::smatch m;
    if (!std::regex_search(name, m, re)) return false;

    auto toYMD = [](const std::string& s, int& y,int& m,int& d){
        y = std::stoi(s.substr(0,4));
        m = std::stoi(s.substr(4,2));
        d = std::stoi(s.substr(6,2));
    };
    toYMD(m[1], out.y1,out.m1,out.d1);
    toYMD(m[2], out.y2,out.m2,out.d2);
    return true;
}

static time_t ymd_to_days(int Y,int M,int D){
    std::tm tm{}; tm.tm_year=Y-1900; tm.tm_mon=M-1; tm.tm_mday=D;
    tm.tm_hour=0; tm.tm_min=0; tm.tm_sec=0;
#ifdef _WIN32
    return _mkgmtime(&tm)/86400;
#else
    return timegm(&tm)/86400;
#endif
}

static bool ranges_overlap_days(time_t a1,time_t a2,time_t b1,time_t b2) {
    return !(a2 < b1 || b2 < a1);
}

std::vector<std::string>
select_ranged_files_overlapping(const fs::path& dir,
                                const std::string& prefix,
                                const std::string& suffix,
                                int Ys,int Ms,int Ds, int Ye,int Me,int De)
{
    std::vector<std::pair<time_t,std::string>> hits;
    const time_t cs = ymd_to_days(Ys,Ms,Ds), ce = ymd_to_days(Ye,Me,De);

    if (!fs::exists(dir)) {
        std::ostringstream oss; oss << "Directory not found: " << dir.string();
        throw std::runtime_error(oss.str());
    }

    for (const auto& de : fs::directory_iterator(dir)) {
        if (!de.is_regular_file()) continue;
        const std::string name = de.path().filename().string();

        if (!prefix.empty() && name.rfind(prefix, 0) != 0) continue;
        if (!suffix.empty() && (name.size() < suffix.size() ||
            name.substr(name.size()-suffix.size()) != suffix)) continue;

        DateRange dr;
        if (!parse_date_range_from_name(name, dr)) continue;

        const time_t fsd = ymd_to_days(dr.y1,dr.m1,dr.d1);
        const time_t fed = ymd_to_days(dr.y2,dr.m2,dr.d2);

        if (ranges_overlap_days(fsd,fed, cs,ce)) {
            hits.emplace_back(fsd, de.path().string());
        }
    }

    std::sort(hits.begin(), hits.end(),
              [](auto& a, auto& b){ return a.first < b.first; });

    std::vector<std::string> out; out.reserve(hits.size());
    for (auto& h : hits) out.push_back(h.second);
    return out;
}

// ─────────── LookupMapper ───────────
LookupMapper::LookupMapper(const std::string& filepath)
    : filepath_(filepath) {}

bool LookupMapper::load() {
    std::ifstream file(filepath_);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filepath_ << std::endl;
        return false;
    }

    std::string line;
    std::getline(file, line); // header

    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string field;
        long long stream;
        int lat, lon;

        std::getline(ss, field, ',');
        stream = std::stoll(field);

        std::getline(ss, field, ',');
        lat = std::stoi(field);

        std::getline(ss, field, ',');
        lon = std::stoi(field);

        stream_map_[stream] = {lat, lon};
    }

    return true;
}

bool LookupMapper::hasStream(long long stream_id) const {
    return stream_map_.find(stream_id) != stream_map_.end();
}

std::pair<int, int> LookupMapper::getLatLon(long long stream_id) const {
    auto it = stream_map_.find(stream_id);
    if (it != stream_map_.end()) {
        return it->second;
    }
    return {-1, -1};
}

size_t LookupMapper::size() const {
    return stream_map_.size();
}

// ─────────── NetCDFLoader ───────────
void NetCDFLoader::checkError(int status, const std::string& operation) {
    if (status != NC_NOERR) {
        throw std::runtime_error(operation + ": " + nc_strerror(status));
    }
}

NetCDFLoader::NetCDFLoader(const std::string& filename, const std::string& varName)
    : ncid(-1), varid(-1), timeSize(0), latSize(0), lonSize(0), fileName(filename), varName(varName) {

    int status = nc_open(filename.c_str(), NC_NOWRITE, &ncid);
    checkError(status, "Opening file " + filename);

    status = nc_inq_varid(ncid, varName.c_str(), &varid);
    if (status != NC_NOERR) {
        nc_close(ncid);
        ncid = -1;
        throw std::runtime_error("Variable " + varName + " not found in file");
    }

    int ndims;
    int dimids[NC_MAX_VAR_DIMS];
    status = nc_inq_var(ncid, varid, NULL, NULL, &ndims, dimids, NULL);
    checkError(status, "Inquiring variable dimensions");

    if (ndims != 3) {
        nc_close(ncid);
        ncid = -1;
        throw std::runtime_error("Expected 3D variable (time, lat, lon), got " + std::to_string(ndims) + "D");
    }

    status = nc_inq_dimlen(ncid, dimids[0], &timeSize);
    checkError(status, "Getting time dimension size");
    status = nc_inq_dimlen(ncid, dimids[1], &latSize);
    checkError(status, "Getting latitude dimension size");
    status = nc_inq_dimlen(ncid, dimids[2], &lonSize);
    checkError(status, "Getting longitude dimension size");
}

NetCDFLoader::~NetCDFLoader() {
    if (ncid >= 0) {
        nc_close(ncid);
        ncid = -1;
    }
}

NetCDFLoader::NetCDFLoader(NetCDFLoader&& other) noexcept
    : ncid(other.ncid), varid(other.varid),
      timeSize(other.timeSize), latSize(other.latSize), lonSize(other.lonSize),
      fileName(std::move(other.fileName)), varName(std::move(other.varName)) {
    other.ncid = -1;
    other.varid = -1;
    other.timeSize = 0;
    other.latSize = 0;
    other.lonSize = 0;
}

NetCDFLoader& NetCDFLoader::operator=(NetCDFLoader&& other) noexcept {
    if (this != &other) {
        if (ncid >= 0) {
            nc_close(ncid);
        }
        ncid = other.ncid;
        varid = other.varid;
        timeSize = other.timeSize;
        latSize = other.latSize;
        lonSize = other.lonSize;
        fileName = std::move(other.fileName);
        varName = std::move(other.varName);
        other.ncid = -1;
        other.varid = -1;
        other.timeSize = 0;
        other.latSize = 0;
        other.lonSize = 0;
    }
    return *this;
}

std::unique_ptr<float[]> NetCDFLoader::loadTimeChunk(size_t startTime, size_t numTimeSteps) {
    if (numTimeSteps == 0) {
        throw std::invalid_argument("Size of time chunk must be greater than zero");
    }
    if (startTime >= timeSize) {
        throw std::out_of_range("Start time index out of range");
    }
    if (startTime + numTimeSteps > timeSize) {
        throw std::out_of_range("Requested time steps exceed available data");
    }

    size_t actualTimeSteps = std::min(numTimeSteps, timeSize - startTime);
    size_t totalElements = actualTimeSteps * latSize * lonSize;

    std::unique_ptr<float[]> data = std::make_unique<float[]>(totalElements);

    size_t start[3] = {startTime, 0, 0};
    size_t count[3] = {actualTimeSteps, latSize, lonSize};

    int status = nc_get_vara_float(ncid, varid, start, count, data.get());
    checkError(status, "Reading variable data");
    return data;
}

float NetCDFLoader::getValueFromChunk(const std::unique_ptr<float[]>& chunkData,
                                     size_t relativeTimeIndex, size_t latIndex, size_t lonIndex,
                                     size_t chunkTimeSize, size_t latSize, size_t lonSize) {
    if (relativeTimeIndex >= chunkTimeSize ||
        latIndex >= latSize ||
        lonIndex >= lonSize) {
        throw std::out_of_range("Chunk indices out of range");
    }
    size_t index = relativeTimeIndex * (latSize * lonSize) + latIndex * lonSize + lonIndex;
    return chunkData[index];
}

bool NetCDFLoader::isDataLoaded() const {
    return ncid >= 0 && timeSize > 0 && latSize > 0 && lonSize > 0;
}

// ─────────── NetCDF helpers, time parsing, etc. ───────────
#ifndef NC_CHECK
#define NC_CHECK(call) do { \
  int _e = (call); \
  if (_e != NC_NOERR) { \
    throw std::runtime_error(std::string("netCDF error: ") + nc_strerror(_e)); \
  } \
} while(0)
#endif

namespace {

struct TimeUnits { double sec_per_unit = 0.0; std::tm origin{}; };

static time_t timegm_utc(std::tm t) {
#ifdef _WIN32
  return _mkgmtime(&t);
#else
  return timegm(&t);
#endif
}

// Permissive "units" parser (accepts date-only, optional time, T/Z/UTC, fractional seconds)
static bool parse_time_units(const std::string& units_in, TimeUnits& out) {
  std::string units = units_in;
  auto ltrim = [](std::string& s){
    s.erase(0, s.find_first_not_of(" \t\r\n"));
  };
  auto rtrim = [](std::string& s){
    s.erase(s.find_last_not_of(" \t\r\n") + 1);
  };
  ltrim(units); rtrim(units);

  std::string lower = units;
  std::transform(lower.begin(), lower.end(), lower.begin(),
                 [](unsigned char c){ return std::tolower(c); });

  const bool is_hours = lower.rfind("hours since",   0) == 0;
  const bool is_days  = lower.rfind("days since",    0) == 0;
  const bool is_min   = lower.rfind("minutes since", 0) == 0;
  const bool is_sec   = lower.rfind("seconds since", 0) == 0;
  if (!(is_hours || is_days || is_min || is_sec)) return false;

  out.sec_per_unit = is_hours ? 3600.0
                    : is_days ? 86400.0
                    : is_min  ? 60.0
                              : 1.0;

  size_t pos = lower.find("since");
  if (pos == std::string::npos) return false;

  std::string ts = units.substr(pos + 5);
  ltrim(ts); rtrim(ts);

  std::replace(ts.begin(), ts.end(), 'T', ' ');
  if (!ts.empty() && (ts.back()=='Z' || ts.back()=='z')) ts.pop_back();
  rtrim(ts);

  auto ends_with_ci = [](const std::string& s, const char* suf){
    const size_t n = std::strlen(suf);
    if (s.size() < n) return false;
    for (size_t i=0;i<n;++i) {
      char a = std::tolower(static_cast<unsigned char>(s[s.size()-n+i]));
      char b = std::tolower(static_cast<unsigned char>(suf[i]));
      if (a != b) return false;
    }
    return true;
  };
  if (ends_with_ci(ts, "utc")) { ts.erase(ts.size()-3); rtrim(ts); }

  int Y=0, M=1, D=1, h=0, m=0;
  double s = 0.0;

  int scanned = std::sscanf(ts.c_str(), "%d-%d-%d %d:%d:%lf", &Y,&M,&D,&h,&m,&s);
  if (scanned < 3) {
    if (std::sscanf(ts.c_str(), "%d-%d-%d", &Y,&M,&D) != 3) return false;
    h = m = 0; s = 0.0;
  } else if (scanned == 3) {
    h = m = 0; s = 0.0;
  } else if (scanned == 5) {
    s = 0.0;
  }

  std::tm tm{};
  tm.tm_year = Y - 1900;
  tm.tm_mon  = M - 1;
  tm.tm_mday = D;
  tm.tm_hour = h;
  tm.tm_min  = m;
  tm.tm_sec  = static_cast<int>(std::lround(s)); // round fractional seconds

  out.origin = tm;
  return true;
}

// Dimension length helper
static size_t dim_len(int ncid, const char* name) {
  int dimid; NC_CHECK(nc_inq_dimid(ncid, name, &dimid));
  size_t n = 0; NC_CHECK(nc_inq_dimlen(ncid, dimid, &n));
  return n;
}

// Boost per-var chunk cache (no-op if not supported)
static void boost_chunk_cache(int ncid, int varid) {
  size_t bytes = size_t(512) * 1024 * 1024; // 512 MiB
  size_t nelems = 1000000;
  float  preempt = 0.75f;
  (void)nc_set_var_chunk_cache(ncid, varid, bytes, nelems, preempt);
}

// Parse ISO-like timestamps to epoch (UTC).
static time_t parse_iso_to_epoch(const std::string& s_in) {
    std::string s = s_in;
    auto b = s.find_first_not_of(" \t\r\n");
    auto e = s.find_last_not_of(" \t\r\n");
    s = (b == std::string::npos) ? std::string() : s.substr(b, e - b + 1);

    std::replace(s.begin(), s.end(), 'T', ' ');
    if (!s.empty() && (s.back()=='Z' || s.back()=='z')) s.pop_back();

    int Y=0,M=1,D=1,h=0,m=0,sec=0;
    if (std::sscanf(s.c_str(), "%d-%d-%d %d:%d:%d", &Y,&M,&D,&h,&m,&sec) >= 3) {
        // ok
    } else if (s.size()==8 && std::all_of(s.begin(), s.end(), ::isdigit)) {
        Y = std::stoi(s.substr(0,4));
        M = std::stoi(s.substr(4,2));
        D = std::stoi(s.substr(6,2));
    } else if (std::sscanf(s.c_str(), "%d-%d-%d", &Y,&M,&D) == 3) {
        // date-only
    } else {
        throw std::runtime_error("Unrecognized time string: " + s_in);
    }

    std::tm tm{}; tm.tm_year = Y-1900; tm.tm_mon = M-1; tm.tm_mday = D;
    tm.tm_hour = h; tm.tm_min = m; tm.tm_sec = sec;
#ifdef _WIN32
    return _mkgmtime(&tm);
#else
    return timegm(&tm);
#endif
}

// Read time axis as epoch seconds for any vartype (numeric or string).
static void read_time_epoch(int ncid, int tvar, size_t ntim,
                            const TimeUnits& tu, bool /*time_is_numeric*/,
                            std::vector<time_t>& tepoch)
{
    tepoch.resize(ntim);

    nc_type xtype;
    NC_CHECK(nc_inq_vartype(ncid, tvar, &xtype));

    if (xtype == NC_STRING) {
        std::vector<char*> arr(ntim);
        NC_CHECK(nc_get_var_string(ncid, tvar, arr.data()));
        for (size_t i=0;i<ntim;++i) {
            tepoch[i] = parse_iso_to_epoch(arr[i] ? std::string(arr[i]) : std::string());
        }
        NC_CHECK(nc_free_string(ntim, arr.data()));
        return;
    }

    if (xtype == NC_CHAR) {
        int ndims = 0; int dimids[NC_MAX_VAR_DIMS];
        NC_CHECK(nc_inq_var(ncid, tvar, nullptr, nullptr, &ndims, dimids, nullptr));
        if (ndims != 2) throw std::runtime_error("char time var must be 2D (time,strlen)");
        size_t ntime=0, nlen=0;
        NC_CHECK(nc_inq_dimlen(ncid, dimids[0], &ntime));
        NC_CHECK(nc_inq_dimlen(ncid, dimids[1], &nlen));
        if (ntime != ntim) throw std::runtime_error("time dimension mismatch for char time var");

        std::vector<char> buf(ntim * nlen);
        NC_CHECK(nc_get_var_text(ncid, tvar, buf.data()));
        for (size_t i=0;i<ntim;++i) {
            const char* row = buf.data() + i*nlen;
            std::string s(row, row + nlen);
            auto last = s.find_last_not_of(std::string(" \0",2));
            if (last != std::string::npos) s.resize(last+1); else s.clear();
            tepoch[i] = parse_iso_to_epoch(s);
        }
        return;
    }

    // numeric
    std::vector<double> tvals(ntim);
    NC_CHECK(nc_get_var_double(ncid, tvar, tvals.data()));
#ifdef _WIN32
    const time_t origin_epoch = _mkgmtime(const_cast<std::tm*>(&tu.origin));
#else
    const time_t origin_epoch = timegm(const_cast<std::tm*>(&tu.origin));
#endif
    for (size_t i=0;i<ntim;++i) {
        tepoch[i] = origin_epoch + static_cast<time_t>(tvals[i] * tu.sec_per_unit);
    }
}

// Read a string attribute (works for NC_CHAR and NC_STRING).
static std::string read_att_string(int ncid, int varid, const char* name) {
    nc_type atype;
    if (nc_inq_atttype(ncid, varid, name, &atype) != NC_NOERR) return "";
    size_t len = 0;
    NC_CHECK(nc_inq_attlen(ncid, varid, name, &len));

    if (atype == NC_CHAR) {
        std::string s(len, '\0');
        NC_CHECK(nc_get_att_text(ncid, varid, name, s.data()));
        if (!s.empty() && s.back() == '\0') s.pop_back();
        return s;
    } else if (atype == NC_STRING) {
        std::vector<char*> arr(len);
        NC_CHECK(nc_get_att_string(ncid, varid, name, arr.data()));
        std::string s;
        if (len > 0 && arr[0]) s = arr[0];
        NC_CHECK(nc_free_string(len, arr.data()));
        return s;
    }
    return "";
}

} // unnamed namespace

// ─────────── Main loader ───────────
void NCForcing::loadData() {
    // --- reset / prepare ---
    h_data.clear();
    nForc = static_cast<int>(entries.size());
    dt.resize(nForc);
    nT.resize(nForc);

    // chunk window [start, end) in epoch seconds (UTC)
    std::tm s_tm{}; s_tm.tm_year = Ys-1900; s_tm.tm_mon = Ms-1; s_tm.tm_mday = Ds;
    std::tm e_tm{}; e_tm.tm_year = Ye-1900; e_tm.tm_mon = Me-1; e_tm.tm_mday = De;
    const time_t chunk_start = timegm_utc(s_tm);          // inclusive 00:00 of start day
    const time_t chunk_end   = timegm_utc(e_tm) + 86400;  // exclusive 00:00 after end day

    // grid dims (from the first opened file)
    size_t nlat_global = 0, nlon_global = 0;
    bool grid_dims_set = false;

    // flattened pixel index per system (iy*nlon + ix)
    std::vector<int> pix_index(systemIds.size(), -1);

    auto build_pix_index = [&](size_t nlon) {
        for (size_t s = 0; s < systemIds.size(); ++s) {
            auto [iy, ix] = mapper.getLatLon(systemIds[s]);
            if (iy < 0 || ix < 0) {
                throw std::runtime_error("Invalid lat/lon mapping for stream " +
                                         std::to_string(systemIds[s]));
            }
            if (static_cast<size_t>(iy) >= nlat_global || static_cast<size_t>(ix) >= nlon) {
                throw std::runtime_error("Lat/Lon mapping out of bounds for stream " +
                                         std::to_string(systemIds[s]));
            }
            pix_index[s] = iy * static_cast<int>(nlon) + ix;
        }
    };

    // ── first pass per forcing: compute nT[f] and set dt[f] ──
    for (int f = 0; f < nForc; ++f) {
        const auto& ent = entries[f];
        dt[f] = ent.dt_hours;
        size_t total_steps = 0;

        for (const auto& path : ent.files) {
            int ncid=-1, varid=-1, tvar=-1;
            NC_CHECK(nc_open(path.c_str(), NC_NOWRITE, &ncid));

            // dims (must be time, latitude, longitude)
            const size_t nlat = dim_len(ncid, ent.dims_lat.c_str());
            const size_t nlon = dim_len(ncid, ent.dims_lon.c_str());
            const size_t ntim = dim_len(ncid, ent.dims_time.c_str());

            if (!grid_dims_set) {
                nlat_global = nlat; nlon_global = nlon; grid_dims_set = true;
                build_pix_index(nlon_global);
            } else if (nlat != nlat_global || nlon != nlon_global) {
                nc_close(ncid);
                throw std::runtime_error("Grid dims mismatch across files for forcing " + ent.var_name);
            }

            NC_CHECK(nc_inq_varid(ncid, ent.var_name.c_str(), &varid));
            NC_CHECK(nc_inq_varid(ncid, ent.dims_time.c_str(), &tvar));
            boost_chunk_cache(ncid, varid); // ok if no-op

            // time vartype
            nc_type time_type;
            NC_CHECK(nc_inq_vartype(ncid, tvar, &time_type));

            TimeUnits tu{};
            bool time_is_numeric = !(time_type == NC_CHAR || time_type == NC_STRING);
            if (time_is_numeric) {
                std::string units = read_att_string(ncid, tvar, "units");
                if (units.empty() || !parse_time_units(units, tu)) {
                    nc_close(ncid);
                    throw std::runtime_error("Unsupported or missing time units on numeric time var: " + units);
                }
            }

            std::vector<time_t> tepoch;
            read_time_epoch(ncid, tvar, ntim, tu, time_is_numeric, tepoch);

            // overlap with chunk
            size_t i0 = 0; while (i0 < ntim && tepoch[i0] < chunk_start) ++i0;
            size_t i1 = i0; while (i1 < ntim && tepoch[i1] < chunk_end)   ++i1;
            if (i0 < i1) total_steps += (i1 - i0);

            nc_close(ncid);
        }
        nT[f] = total_steps;
    }

    // ── allocate final buffer [f][t][system] ──
    const size_t total_elems = [&]{
        size_t s=0; for (int f=0; f<nForc; ++f) s += nT[f] * size_t(systems);
        return s;
    }();
    h_data.assign(total_elems, 0.0f);

    auto baseF = [&](int f)->size_t {
        size_t base = 0;
        for (int k = 0; k < f; ++k) base += nT[k] * size_t(systems);
        return base;
    };

    // host slice buffer [lat,lon]
    std::vector<float> slice_host;
    if (nlat_global == 0 || nlon_global == 0) {
        // no data available across all forcings
        return;
    }
    slice_host.resize(nlat_global * nlon_global);

    // ── second pass: read [time][lat][lon] slices and scatter to [f][t][system] ──
    for (int f = 0; f < nForc; ++f) {
        const auto& ent = entries[f];
        const size_t Fbase = baseF(f);
        size_t t_written = 0;

        for (const auto& path : ent.files) {
            int ncid=-1, varid=-1, tvar=-1;
            NC_CHECK(nc_open(path.c_str(), NC_NOWRITE, &ncid));
            NC_CHECK(nc_inq_varid(ncid, ent.var_name.c_str(), &varid));
            NC_CHECK(nc_inq_varid(ncid, ent.dims_time.c_str(), &tvar));
            boost_chunk_cache(ncid, varid);

            // ensure data var numeric
            nc_type vtype;
            NC_CHECK(nc_inq_vartype(ncid, varid, &vtype));
            if (vtype == NC_CHAR || vtype == NC_STRING) {
                nc_close(ncid);
                throw std::runtime_error(
                    "Variable '" + ent.var_name + "' is text (CHAR/STRING). Check var name; cannot read as float.");
            }

            const size_t ntim = dim_len(ncid, ent.dims_time.c_str());

            nc_type time_type;
            NC_CHECK(nc_inq_vartype(ncid, tvar, &time_type));

            TimeUnits tu{};
            bool time_is_numeric = !(time_type == NC_CHAR || time_type == NC_STRING);
            if (time_is_numeric) {
                std::string units = read_att_string(ncid, tvar, "units");
                if (units.empty() || !parse_time_units(units, tu)) {
                    nc_close(ncid);
                    throw std::runtime_error("Unsupported or missing time units on numeric time var: " + units);
                }
            }

            std::vector<time_t> tepoch;
            read_time_epoch(ncid, tvar, ntim, tu, time_is_numeric, tepoch);

            // find [i0,i1) overlap for this file
            size_t i0 = 0; while (i0 < ntim && tepoch[i0] < chunk_start) ++i0;
            size_t i1 = i0; while (i1 < ntim && tepoch[i1] < chunk_end)   ++i1;
            if (i0 >= i1) { nc_close(ncid); continue; }

            size_t start[3] = {0, 0, 0};
            size_t count[3] = {1, nlat_global, nlon_global};

            for (size_t ti = i0; ti < i1; ++ti) {
                start[0] = ti;
                NC_CHECK(nc_get_vara_float(ncid, varid, start, count, slice_host.data()));

                float* dst = h_data.data() + (Fbase + t_written * size_t(systems));
                for (size_t s = 0; s < size_t(systems); ++s) {
                    dst[s] = slice_host[ static_cast<size_t>(pix_index[s]) ];
                }
                ++t_written;
            }

            nc_close(ncid);
        }

        if (t_written != nT[f]) {
            throw std::runtime_error("t_written != nT for forcing " + std::to_string(f));
        }

        if (nT[f] > 0) {
            float first = h_data[Fbase + 0 * size_t(systems) + 0];
            float last  = h_data[Fbase + (nT[f]-1) * size_t(systems) + 0];
            std::cout << "[FORCING DEBUG] f=" << f
                      << " total_T=" << nT[f]
                      << " first=" << first
                      << " last="  << last << "\n";
        } else {
            std::cout << "[FORCING DEBUG] f=" << f << " produced no data (check overlap)\n";
        }
    }
}
