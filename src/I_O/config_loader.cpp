#include "config_loader.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cctype>

// Implementation of SimpleYamlParser methods
void SimpleYamlParser::parseFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(file, line)) {
        lines.push_back(line);
    }
    file.close();
    
    parseLines(lines);
}

std::string SimpleYamlParser::trim(const std::string& str) {
    size_t start = str.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) return "";
    size_t end = str.find_last_not_of(" \t\n\r");
    return str.substr(start, end - start + 1);
}

std::string SimpleYamlParser::removeQuotes(const std::string& str) {
    std::string trimmed = trim(str);
    
    // Remove inline comments first
    size_t commentPos = trimmed.find('#');
    if (commentPos != std::string::npos) {
        trimmed = trim(trimmed.substr(0, commentPos));
    }
    
    // Then remove quotes
    if (!trimmed.empty() && 
        ((trimmed.front() == '"' && trimmed.back() == '"') ||
         (trimmed.front() == '\'' && trimmed.back() == '\''))) {
        return trimmed.substr(1, trimmed.length() - 2);
    }
    return trimmed;
}

bool SimpleYamlParser::isInlineArray(const std::string& str) {
    std::string trimmed = trim(str);
    return !trimmed.empty() && trimmed.front() == '[' && trimmed.back() == ']';
}

std::vector<std::string> SimpleYamlParser::parseInlineArray(const std::string& str) {
    std::vector<std::string> result;
    std::string trimmed = trim(str);
    
    if (!isInlineArray(trimmed)) {
        return result;
    }
    
    // Remove brackets
    std::string content = trimmed.substr(1, trimmed.length() - 2);
    
    // Split by comma
    std::stringstream ss(content);
    std::string item;
    
    while (std::getline(ss, item, ',')) {
        std::string cleanItem = trim(item);
        if (!cleanItem.empty()) {
            result.push_back(removeQuotes(cleanItem));
        }
    }
    
    return result;
}

int SimpleYamlParser::getIndentLevel(const std::string& line) {
    int indent = 0;
    for (char c : line) {
        if (c == ' ') indent++;
        else if (c == '\t') indent += 4; // Treat tab as 4 spaces
        else break;
    }
    return indent;
}

bool SimpleYamlParser::isArrayItem(const std::string& line) {
    std::string trimmed = trim(line);
    return !trimmed.empty() && trimmed[0] == '-';
}

bool SimpleYamlParser::isComment(const std::string& line) {
    std::string trimmed = trim(line);
    return trimmed.empty() || trimmed[0] == '#';
}

void SimpleYamlParser::parseLines(const std::vector<std::string>& lines) {
    std::vector<std::string> sectionPath;
    std::string currentArrayKey;
    std::vector<std::map<std::string, std::string>> currentArray;
    std::vector<std::string> currentSimpleArray;
    bool inArray = false;
    bool inSimpleArray = false;
    int baseIndent = -1;
    
    for (size_t i = 0; i < lines.size(); i++) {
        const std::string& line = lines[i];
        
        if (isComment(line)) continue;
        
        int indent = getIndentLevel(line);
        std::string trimmed = trim(line);
        
        if (trimmed.empty()) continue;
        
        // Handle array items
        if (isArrayItem(trimmed)) {
            std::string arrayContent = trim(trimmed.substr(1)); // Remove '-'
            
            // Check if this is a simple array (just values) or complex array (objects)
            if (arrayContent.find(':') != std::string::npos) {
                // Complex array item (object)
                if (!inArray) {
                    inArray = true;
                    inSimpleArray = false;
                    currentArray.clear();
                }
                
                std::map<std::string, std::string> arrayItem;
                
                // Parse the current line
                size_t colonPos = arrayContent.find(':');
                if (colonPos != std::string::npos) {
                    std::string key = trim(arrayContent.substr(0, colonPos));
                    std::string value = removeQuotes(trim(arrayContent.substr(colonPos + 1)));
                    arrayItem[key] = value;
                }
                
                // Look ahead for more properties of this array item
                for (size_t j = i + 1; j < lines.size(); j++) {
                    std::string nextLine = lines[j];
                    if (isComment(nextLine)) continue;
                    
                    int nextIndent = getIndentLevel(nextLine);
                    std::string nextTrimmed = trim(nextLine);
                    
                    if (nextTrimmed.empty()) continue;
                    
                    // If next line is less indented or another array item, stop
                    if (nextIndent <= indent || isArrayItem(nextTrimmed)) {
                        break;
                    }
                    
                    // Parse key-value pair
                    size_t nextColonPos = nextTrimmed.find(':');
                    if (nextColonPos != std::string::npos) {
                        std::string nextKey = trim(nextTrimmed.substr(0, nextColonPos));
                        std::string nextValue = removeQuotes(trim(nextTrimmed.substr(nextColonPos + 1)));
                        arrayItem[nextKey] = nextValue;
                        i = j; // Skip these lines in main loop
                    }
                }
                
                currentArray.push_back(arrayItem);
            } else {
                // Simple array item (just a value)
                if (!inSimpleArray) {
                    inSimpleArray = true;
                    inArray = false;
                    currentSimpleArray.clear();
                }
                currentSimpleArray.push_back(removeQuotes(arrayContent));
            }
            continue;
        }
        
        // If we were in an array and now we're not, save it
        if ((inArray || inSimpleArray) && !isArrayItem(trimmed)) {
            if (inArray && !currentArray.empty()) {
                arrayMap[currentArrayKey] = currentArray;
            } else if (inSimpleArray && !currentSimpleArray.empty()) {
                simpleArrayMap[currentArrayKey] = currentSimpleArray;
            }
            inArray = false;
            inSimpleArray = false;
            currentArray.clear();
            currentSimpleArray.clear();
        }
        
        // Handle regular key-value pairs
        size_t colonPos = trimmed.find(':');
        if (colonPos != std::string::npos) {
            std::string key = trim(trimmed.substr(0, colonPos));
            std::string value = trim(trimmed.substr(colonPos + 1));
            
            // Remove inline comments from value before checking if it's empty
            std::string cleanValue = value;
            size_t commentPos = cleanValue.find('#');
            if (commentPos != std::string::npos) {
                cleanValue = trim(cleanValue.substr(0, commentPos));
            }
            
            // Adjust section path based on indentation
            if (baseIndent == -1) baseIndent = indent;
            
            int level = (indent - baseIndent) / 2; // Assuming 2-space indentation
            if (level < 0) level = 0;
            
            // Adjust section path
            if (level < sectionPath.size()) {
                sectionPath.resize(level);
            }
            
            if (cleanValue.empty()) {
                // This is a section header
                sectionPath.push_back(key);
                currentArrayKey = getSectionKey(sectionPath);
            } else {
                // This is a key-value pair
                std::vector<std::string> fullPath = sectionPath;
                fullPath.push_back(key);
                std::string fullKey = getSectionKey(fullPath);
              
                // Check if the value is an inline array — USE cleanValue, not value
                if (isInlineArray(cleanValue)) {
                    std::vector<std::string> arrayItems = parseInlineArray(cleanValue);
                    simpleArrayMap[fullKey] = arrayItems;
                } else {
                    keyValueMap[fullKey] = removeQuotes(cleanValue);
                }

            }
        }
    }
    
    // Save any remaining array
    if (inArray && !currentArray.empty()) {
        arrayMap[currentArrayKey] = currentArray;
    } else if (inSimpleArray && !currentSimpleArray.empty()) {
        simpleArrayMap[currentArrayKey] = currentSimpleArray;
    }
}

std::string SimpleYamlParser::getSectionKey(const std::vector<std::string>& path) {
    std::string result;
    for (size_t i = 0; i < path.size(); i++) {
        if (i > 0) result += ".";
        result += path[i];
    }
    return result;
}

std::string SimpleYamlParser::getString(const std::string& key, const std::string& defaultValue) {
    auto it = keyValueMap.find(key);
    return (it != keyValueMap.end()) ? it->second : defaultValue;
}

int SimpleYamlParser::getInt(const std::string& key, int defaultValue) {
    auto it = keyValueMap.find(key);
    if (it != keyValueMap.end()) {
        return std::stoi(it->second);
    }
    return defaultValue;
}

double SimpleYamlParser::getDouble(const std::string& key, double defaultValue) {
    auto it = keyValueMap.find(key);
    if (it != keyValueMap.end()) {
        return std::stod(it->second);
    }
    return defaultValue;
}

bool SimpleYamlParser::getBool(const std::string& key, bool defaultValue) {
    auto it = keyValueMap.find(key);
    if (it != keyValueMap.end()) {
        std::string value = it->second;
        std::transform(value.begin(), value.end(), value.begin(), ::tolower);
        return value == "true" || value == "yes" || value == "1";
    }
    return defaultValue;
}

std::vector<int> SimpleYamlParser::getIntArray(const std::string& key) {
    std::vector<int> result;
    auto it = simpleArrayMap.find(key);
    if (it != simpleArrayMap.end()) {
        for (const std::string& str : it->second) {
            result.push_back(std::stoi(str));
        }
    }
    return result;
}

std::vector<double> SimpleYamlParser::getDoubleArray(const std::string& key) {
    std::vector<double> result;
    auto it = simpleArrayMap.find(key);
    if (it != simpleArrayMap.end()) {
        for (const std::string& str : it->second) {
            result.push_back(std::stod(str));
        }
    }
    return result;
}

std::vector<std::map<std::string, std::string>> SimpleYamlParser::getObjectArray(const std::string& key) {
    auto it = arrayMap.find(key);
    return (it != arrayMap.end()) ? it->second : std::vector<std::map<std::string, std::string>>();
}

void SimpleYamlParser::printParsedData() {
    std::cout << "=== Key-Value Pairs ===" << std::endl;
    for (const auto& pair : keyValueMap) {
        std::cout << pair.first << " = " << pair.second << std::endl;
    }
    
    std::cout << "\n=== Simple Arrays ===" << std::endl;
    for (const auto& pair : simpleArrayMap) {
        std::cout << pair.first << " = [";
        for (size_t i = 0; i < pair.second.size(); i++) {
            if (i > 0) std::cout << ", ";
            std::cout << pair.second[i];
        }
        std::cout << "]" << std::endl;
    }
    
    std::cout << "\n=== Object Arrays ===" << std::endl;
    for (const auto& pair : arrayMap) {
        std::cout << pair.first << " = [" << std::endl;
        for (const auto& obj : pair.second) {
            std::cout << "  {";
            bool first = true;
            for (const auto& objPair : obj) {
                if (!first) std::cout << ", ";
                std::cout << objPair.first << ": " << objPair.second;
                first = false;
            }
            std::cout << "}" << std::endl;
        }
        std::cout << "]" << std::endl;
    }
}

// Implementation of ConfigLoader methods
ModelConfig ConfigLoader::loadConfig(const std::string& filename) {
    SimpleYamlParser parser;
    parser.parseFile(filename);
    
    ModelConfig config;

    // Load general information section
    config.description = parser.getString("description");
    
    // Load model section
    config.model_uid = parser.getInt("model.uid");
    config.model_name = parser.getString("model.name");
    
    // Load time period
    config.time_start = parser.getString("time_period.start");
    config.time_end = parser.getString("time_period.end");
    
    // Load initial section
    config.initial_mode = parser.getString("initial.mode");
    config.initial_file = parser.getString("initial.file");
    config.initial_values = parser.getDoubleArray("initial.values");

    // Validate size if provided
    // state size = 9 for now, change if using a different model
    constexpr int STATE_SIZE = 9;
    if (!config.initial_values.empty() && 
        static_cast<int>(config.initial_values.size()) != STATE_SIZE) {
        throw std::runtime_error(
            "Initial values must have exactly " + std::to_string(STATE_SIZE) + " entries");
    }

    
    // Load parameters
    config.parameters_path = parser.getString("parameters.path");
    config.spatially_varying_file = parser.getString("parameters.spatially_varying_file");
    config.constant_parameters_index = parser.getIntArray("parameters.constant_parameters_index");
    config.constant_parameters_values = parser.getDoubleArray("parameters.constant_parameters_values");

    // Load forcings
    config.forcings_type = parser.getString("forcings.type");
    std::transform(config.forcings_type.begin(), config.forcings_type.end(),
                config.forcings_type.begin(), ::tolower);

    // Default to "yearly" if omitted / unexpected
    if (config.forcings_type != "yearly" && config.forcings_type != "ranged") {
        config.forcings_type = "yearly";
    }

    config.forcings_path = parser.getString("forcings.path");
    config.time_chunking = parser.getBool("forcings.time_chunking");

    // Load forcing variables
    auto forcingVars = parser.getObjectArray("forcings.variables");
    for (const auto& varMap : forcingVars) {
        ForcingVariable var;

        // required fields
        var.name            = varMap.at("name");
        var.var_name        = varMap.at("var_name");
        var.time_resolution = varMap.at("time_resolution");
        var.required        = (varMap.at("required") == "true");

        // optional time_chunk_size
        if (auto it = varMap.find("time_chunk_size"); it != varMap.end()) {
            var.time_chunk_size = std::stoi(it->second);
        } else {
            var.time_chunk_size = -1;
        }

        // yearly vs ranged-by-name (driven by forcings.type)
        var.ranged_by_name = (config.forcings_type == "ranged");

        // read optional fields
        if (auto it = varMap.find("file"); it != varMap.end()) {
            var.file = it->second;
        }
        if (auto it = varMap.find("file_pattern"); it != varMap.end()) {
            var.file_pattern = it->second;
        }

        // Accept `file:` as alias for `file_pattern:` in ranged mode
        if (var.ranged_by_name && var.file_pattern.empty()) {
            if (!var.file.empty()) {
                // Only adopt if it actually looks like a ranged template
                if (var.file.find("{start}") != std::string::npos &&
                    var.file.find("{end}")   != std::string::npos) {
                    var.file_pattern = var.file;
                }
            }
        }

        // NOTE: current SimpleYamlParser does not capture nested arrays per-variable (e.g., `files:`).
        // We rely on file_pattern (or file mapped to file_pattern) in ranged mode.
             
        // Dimension names: defaults match AORC (time, latitude, longitude)
        // Users can override with dims_time/dims_lat/dims_lon or with
        // a shorthand: dims: "time, lat, lon".
        auto trim_ws = [](std::string s) {
            size_t b = s.find_first_not_of(" \t\r\n");
            if (b == std::string::npos) return std::string();
            size_t e = s.find_last_not_of(" \t\r\n");
            return s.substr(b, e - b + 1);
        };

        std::string d_time = "time";
        std::string d_lat  = "latitude";
        std::string d_lon  = "longitude";

        if (auto it = varMap.find("dims_time"); it != varMap.end() && !it->second.empty())
            d_time = it->second;
        if (auto it = varMap.find("dims_lat"); it != varMap.end() && !it->second.empty())
            d_lat = it->second;
        if (auto it = varMap.find("dims_lon"); it != varMap.end() && !it->second.empty())
            d_lon = it->second;

        if (auto it = varMap.find("dims"); it != varMap.end() && !it->second.empty()) {
            std::vector<std::string> toks;
            std::stringstream ss(it->second);
            std::string tok;
            while (std::getline(ss, tok, ',')) toks.push_back(trim_ws(tok));
            if (toks.size() == 3) {
                if (!toks[0].empty()) d_time = toks[0];
                if (!toks[1].empty()) d_lat  = toks[1];
                if (!toks[2].empty()) d_lon  = toks[2];
            } else {
                throw std::runtime_error(
                    "For forcing '" + var.name + "': 'dims' must have exactly 3 comma-separated names (time, lat, lon).");
            }
        }

        var.dims_time = d_time;
        var.dims_lat  = d_lat;
        var.dims_lon  = d_lon;


        config.forcing_variables.push_back(var);
    }

    // Validate/normalize forcing vars
    for (auto& v : config.forcing_variables) {
        if (v.ranged_by_name) {
            // Must have a pattern with {start} and {end}
            if (v.file_pattern.empty()) {
                throw std::runtime_error(
                    "For ranged forcing '" + v.name +
                    "' provide 'file' or 'file_pattern' containing {start} and {end}.");
            }
            if (v.file_pattern.find("{start}") == std::string::npos ||
                v.file_pattern.find("{end}")   == std::string::npos) {
                throw std::runtime_error(
                    "Ranged forcing '" + v.name +
                    "': pattern must include both {start} and {end}.");
            }
        } else {
            // yearly
            if (v.file.empty()) {
                throw std::runtime_error(
                    "For yearly forcing '" + v.name +
                    "' please provide 'file' with {year}.");
            }
            if (v.file.find("{year}") == std::string::npos) {
                throw std::runtime_error(
                    "Yearly forcing '" + v.name +
                    "': 'file' must include {year}.");
            }
        }
    }

    // Load forcing mappings
    config.forcing_mappings_path = parser.getString("forcing_mappings.path");
    auto mappingVars = parser.getObjectArray("forcing_mappings.variables");
    for (const auto& mappingMap : mappingVars) {
        ForcingMapping mapping;
        mapping.name = mappingMap.at("name");
        mapping.file = mappingMap.at("file");
        config.forcing_mappings.push_back(mapping);
    }
    
    // Load output
    config.print_interval = parser.getInt("output.print_interval");
    // time step for query output in minutes (dense / runoff sampling)
    config.query_dt_minutes = parser.getDouble("output.query_dt", 60.0); // Default to 60 minutes if not specified
    // controls how often to write 2D "final" snapshot files (system × variable)
    config.final_interval_minutes = parser.getDouble("output.final_interval_minutes", 0.0);
    config.output_states = parser.getIntArray("output.states");
    config.output_path = parser.getString("output.output_path");
    config.output_file = parser.getString("output.output_file");
    config.final_output_file = parser.getString("output.final_output_file");
    config.runoff_output_file = parser.getString("output.runoff_output_file");
    config.final_per_year = parser.getBool("output.final_per_year", false);

    // Solver (provide defaults = current hard-coded values)
    config.rtol                 = parser.getDouble("solver.rtol",        1e-6);
    config.atol                 = parser.getDouble("solver.atol",        1e-9);
    config.safety               = parser.getDouble("solver.safety",      0.9);
    if (config.safety > 1.0) {
        std::cerr << "Warning: solver.safety (" << config.safety
                << ") is greater than 1.0. Clamping to 1.0." << std::endl;
        config.safety = 1.0;
    }
    config.min_scale            = parser.getDouble("solver.min_scale",   0.2);
    config.max_scale            = parser.getDouble("solver.max_scale",   10.0);
    config.initial_step         = parser.getDouble("solver.initial_step",0.01);

    // Override flags default to false
    config.override_tolerances  = parser.getBool("solver.override_tolerances",  false);
    config.override_initial_step= parser.getBool("solver.override_initial_step",false);
    
    // Load flags
    config.use_mpi = parser.getBool("flags.use_mpi", false);
    config.max_gpu_mem_gb     = parser.getDouble("flags.max_gpu_mem_gb", 15.0); // Default to 15 GiB
    config.gpu_mem_buffer_pct = parser.getDouble("flags.gpu_mem_buffer_pct", 5.0); // Default to 5%
    if (config.max_gpu_mem_gb <= 0.0) {
        throw std::runtime_error(
            "You must set flags.max_gpu_mem_gb > 0 in your config YAML (GiB)."
        );
    }
    
    return config;
}
