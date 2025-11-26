#ifndef CONFIG_LOADER_HPP
#define CONFIG_LOADER_HPP

#include <string>
#include <vector>
#include <map>

// Structure to hold forcing variable configuration
struct ForcingVariable {
    std::string name;
    std::string file;                 // yearly mode: e.g., "AORC_t2m_daily_avg_{year}_CONUS.nc"
    std::string var_name;
    std::string time_resolution;
    int         time_chunk_size;
    bool        required;

    // ── Ranged files ───────────────────────────────────────────
    // If using ranged mode you can either:
    //  (a) supply an explicit list of files, OR
    //  (b) supply a pattern to be expanded elsewhere.
    std::vector<std::string> files;   // explicit list of files for this variable (ranged mode)
    std::string file_pattern;         // optional pattern with {start} and {end}, e.g.
                                      // "AORC_t2m_daily_avg_{start}_{end}_CONUS.nc"
    bool ranged_by_name = false;      // true → option 2 (date-ranged filenames); false → option 1 (yearly)
    // Mandatory dims (parsed from "dims: 'time, lat, lon'")
    // we keep split fields to pass down to loader
    std::string dims_time;            // e.g. "valid_time" or "time"
    std::string dims_lat;             // e.g. "latitude"
    std::string dims_lon;             // e.g. "longitude"
};

// Structure to hold forcing mapping configuration
struct ForcingMapping {
    std::string name;
    std::string file;
};

// Main configuration structure
struct ModelConfig {
    // General information
    std::string description;

    // Model section
    int         model_uid;
    std::string model_name;
    
    // Time period
    std::string time_start;
    std::string time_end;
    
    // Initial conditions
    std::string initial_mode;
    std::string initial_file;
    std::vector<double> initial_values; // used if initial_mode == "constant"
    
    // Parameters
    std::string              parameters_path;
    std::string              spatially_varying_file;
    std::vector<int>         constant_parameters_index;
    std::vector<double>      constant_parameters_values;
    
    // Forcings
    std::string              forcings_type;   // "yearly" or "ranged"  (use to set ranged_by_name)
    std::string              forcings_path;
    bool                     time_chunking;
    std::vector<ForcingVariable> forcing_variables;
    
    // Forcing mappings
    std::string              forcing_mappings_path;
    std::vector<ForcingMapping> forcing_mappings;
    
    // Output
    int                      print_interval;     // time step for output in minutes
    double                   query_dt_minutes;   // time step for query output in minutes
    std::vector<int>         output_states;      // list of state indices to output
    std::string              output_path;
    std::string              output_file;
    std::string              final_output_file;
    std::string              runoff_output_file;  
    bool                     final_per_year;      // write final only once per year if true
    double                   final_interval_minutes; // minutes between final snapshot files (<=0 disables)
    
    // Solver
    double rtol;                
    double atol;                
    double safety;              
    double min_scale;           
    double max_scale;           
    bool   override_tolerances; 
    double initial_step;        
    bool   override_initial_step;
    
    // Flags
    bool use_mpi;  // Use MPI for parallel execution
    double max_gpu_mem_gb;       // GPU memory budget (GiB)
    double gpu_mem_buffer_pct;   // Reserve percentage of GPU memory for overhead
};

// Simple YAML parser class
class SimpleYamlParser {
public:
    // Public interface
    void parseFile(const std::string& filename);
    
    // Getter methods
    std::string getString(const std::string& key, const std::string& defaultValue = "");
    int         getInt(const std::string& key, int defaultValue = 0);
    double      getDouble(const std::string& key, double defaultValue = 0.0);
    bool        getBool(const std::string& key, bool defaultValue = false);
    std::vector<int>    getIntArray(const std::string& key);
    std::vector<double> getDoubleArray(const std::string& key);
    std::vector<std::map<std::string, std::string>> getObjectArray(const std::string& key);
    void printParsedData();

private:
    // Member variables
    std::map<std::string, std::string>                                     keyValueMap;
    std::map<std::string, std::vector<std::map<std::string, std::string>>> arrayMap;
    std::map<std::string, std::vector<std::string>>                        simpleArrayMap;
    
    // Private helper methods
    void        parseLines(const std::vector<std::string>& lines);
    std::string getSectionKey(const std::vector<std::string>& path);
    
    // Static utility functions
    static std::string trim(const std::string& str);
    static std::string removeQuotes(const std::string& str);
    static bool        isInlineArray(const std::string& str);
    static std::vector<std::string> parseInlineArray(const std::string& str);
    static int         getIndentLevel(const std::string& line);
    static bool        isArrayItem(const std::string& line);
    static bool        isComment(const std::string& line);
};

// Configuration loader class
class ConfigLoader {
public:
    static ModelConfig loadConfig(const std::string& filename);
};

#endif // CONFIG_LOADER_HPP
