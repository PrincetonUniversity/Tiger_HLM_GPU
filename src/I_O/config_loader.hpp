#ifndef CONFIG_LOADER_HPP
#define CONFIG_LOADER_HPP

#include <string>
#include <vector>
#include <map>

// Structure to hold forcing variable configuration
struct ForcingVariable {
    std::string name;
    std::string file;
    std::string var_name;
    std::string time_resolution;
    int time_chunk_size;
    bool required;
};

// Structure to hold forcing mapping configuration
struct ForcingMapping {
    std::string name;
    std::string file;
};

// Main configuration structure
struct ModelConfig {
    // Model section
    int model_uid;
    std::string model_name;
    
    // Time period
    std::string time_start;
    std::string time_end;
    
    // Initial conditions
    std::string initial_mode;
    std::string initial_file;
    
    // Parameters
    std::string parameters_path;
    std::string spatially_varying_file;
    std::vector<int> constant_parameters_index;
    std::vector<double> constant_parameters_values;
    
    // Forcings
    std::string forcings_type;
    std::string forcings_path;
    bool time_chunking;
    std::vector<ForcingVariable> forcing_variables;
    
    // Forcing mappings
    std::string forcing_mappings_path;
    std::vector<ForcingMapping> forcing_mappings;
    
    // Output
    int print_interval;
    std::vector<int> output_states;
    std::string output_path;
    std::string output_file;
    bool final_output;
    std::string final_output_file;
    
    // Solver
    double rtol;
    double atol;
    double safety;
    double min_scale;
    double max_scale;
    bool override_tolerances;
    double initial_step;
    bool override_initial_step;
    
    // MPI
    int step_storage;
    int transfer_buffer;
    int discontinuity_buf;
    
    // Flags
    bool uses_dam;
    bool convert_area;
};

// Simple YAML parser class
class SimpleYamlParser {
public:
    // Public interface
    void parseFile(const std::string& filename);
    
    // Getter methods
    std::string getString(const std::string& key, const std::string& defaultValue = "");
    int getInt(const std::string& key, int defaultValue = 0);
    double getDouble(const std::string& key, double defaultValue = 0.0);
    bool getBool(const std::string& key, bool defaultValue = false);
    std::vector<int> getIntArray(const std::string& key);
    std::vector<double> getDoubleArray(const std::string& key);
    std::vector<std::map<std::string, std::string>> getObjectArray(const std::string& key);
    void printParsedData();

private:
    // Member variables
    std::map<std::string, std::string> keyValueMap;
    std::map<std::string, std::vector<std::map<std::string, std::string>>> arrayMap;
    std::map<std::string, std::vector<std::string>> simpleArrayMap;
    
    // Private helper methods
    void parseLines(const std::vector<std::string>& lines);
    std::string getSectionKey(const std::vector<std::string>& path);
    
    // Static utility functions
    static std::string trim(const std::string& str);
    static std::string removeQuotes(const std::string& str);
    static bool isInlineArray(const std::string& str);
    static std::vector<std::string> parseInlineArray(const std::string& str);
    static int getIndentLevel(const std::string& line);
    static bool isArrayItem(const std::string& line);
    static bool isComment(const std::string& line);
};

// Configuration loader class
class ConfigLoader {
public:
    static ModelConfig loadConfig(const std::string& filename);
};

#endif // CONFIG_LOADER_HPP