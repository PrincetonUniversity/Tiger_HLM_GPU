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
                
                // Check if the value is an inline array
                if (isInlineArray(value)) {
                    std::vector<std::string> arrayItems = parseInlineArray(value);
                    simpleArrayMap[fullKey] = arrayItems;
                } else {
                    keyValueMap[fullKey] = removeQuotes(value);
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
    
    // Load model section
    config.model_uid = parser.getInt("model.uid");
    config.model_name = parser.getString("model.name");
    
    // Load time period
    config.time_start = parser.getString("time_period.start");
    config.time_end = parser.getString("time_period.end");
    
    // Load initial section
    config.initial_mode = parser.getString("initial.mode");
    config.initial_file = parser.getString("initial.file");
    
    // Load parameters
    config.parameters_path = parser.getString("parameters.path");
    config.spatially_varying_file = parser.getString("parameters.spatially_varying_file");
    config.constant_parameters_index = parser.getIntArray("parameters.constant_parameters_index");
    config.constant_parameters_values = parser.getDoubleArray("parameters.constant_parameters_values");
    
    // Load forcings
    config.forcings_type = parser.getString("forcings.type");
    config.forcings_path = parser.getString("forcings.path");
    config.time_chunking = parser.getBool("forcings.time_chunking");
    
    // Load forcing variables
    auto forcingVars = parser.getObjectArray("forcings.variables");
    for (const auto& varMap : forcingVars) {
        ForcingVariable var;
        var.name = varMap.at("name");
        var.file = varMap.at("file");
        var.var_name = varMap.at("var_name");
        var.time_resolution = varMap.at("time_resolution");
        var.time_chunk_size = std::stoi(varMap.at("time_chunk_size"));
        var.required = (varMap.at("required") == "true");
        config.forcing_variables.push_back(var);
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
    config.output_states = parser.getIntArray("output.states");
    config.output_path = parser.getString("output.output_path");
    config.output_file = parser.getString("output.output_file");
    config.final_output = parser.getBool("output.final_output");
    config.final_output_file = parser.getString("output.final_output_file");
    
    // Load solver
    config.rtol = parser.getDouble("solver.rtol");
    config.atol = parser.getDouble("solver.atol");
    config.safety = parser.getDouble("solver.safety");
    config.min_scale = parser.getDouble("solver.min_scale");
    config.max_scale = parser.getDouble("solver.max_scale");
    config.override_tolerances = parser.getBool("solver.override_tolerances");
    config.initial_step = parser.getDouble("solver.initial_step");
    config.override_initial_step = parser.getBool("solver.override_initial_step");
    
    // Load MPI
    config.step_storage = parser.getInt("mpi.step_storage");
    config.transfer_buffer = parser.getInt("mpi.transfer_buffer");
    config.discontinuity_buf = parser.getInt("mpi.discontinuity_buf");
    
    // Load flags
    config.uses_dam = parser.getBool("flags.uses_dam");
    config.convert_area = parser.getBool("flags.convert_area");
    
    return config;
}

// // Example usage
// int main() {
//     try {
//         // Load configuration from file
//         ModelConfig config = ConfigLoader::loadConfig("../../data/config.yaml");
        
//         // Print loaded configuration
//         std::cout << "=== Loaded Configuration ===" << std::endl;

//         std::cout << " -- model -- " << std::endl;
//         std::cout << "Model UID: " << config.model_uid << std::endl;
//         std::cout << "Model Name: " << config.model_name << std::endl;

//         std::cout << " -- time_period -- " << std::endl;
//         std::cout << "Time Period: " << config.time_start << " to " << config.time_end << std::endl;

//         std::cout << " -- initial -- " << std::endl;
//         std::cout << "Initial Mode: " << config.initial_mode << std::endl;
//         std::cout << "Initial File: " << config.initial_file << std::endl;

//         std::cout << " -- parameters -- " << std::endl;
//         std::cout << "Parameters Path: " << config.parameters_path << std::endl;
//         std::cout << "Spatially Varying File: " << config.spatially_varying_file << std::endl;
//         std::cout << "Constant Parameters Index: " << config.constant_parameters_index.size() << std::endl;
//         for (const auto& index : config.constant_parameters_index) {
//             std::cout << "  - Index: " << index << std::endl;
//         }
//         std::cout << "Constant Parameters Values: " << config.constant_parameters_values.size() << std::endl;
//         for (const auto& value : config.constant_parameters_values) {
//             std::cout << "  - Value: " << value << std::endl;
//         }

//         std::cout << " -- forcings -- " << std::endl;
//         std::cout << "Forcings Type: " << config.forcings_type << std::endl;
//         std::cout << "Forcings Path: " << config.forcings_path << std::endl;
//         std::cout << "Time Chunking: " << config.time_chunking << std::endl;
//         std::cout << "Forcing Variables: " << config.forcing_variables.size() << std::endl;
//         for (const auto& var : config.forcing_variables) {
//             std::cout << "  - Name: " << var.name 
//                       << ", File: " << var.file 
//                       << ", Var Name: " << var.var_name 
//                       << ", Time Resolution: " << var.time_resolution 
//                       << ", Chunk Size: " << var.time_chunk_size 
//                       << ", Required: " << (var.required) 
//                       << std::endl;
//         }

//         std::cout << " -- forcing_mappings -- " << std::endl;
//         std::cout << "Forcing Mappings Path: " << config.forcing_mappings_path << std::endl;
//         std::cout << "Forcing Mapping Variables: " << config.forcing_mappings.size() << std::endl;
//         for (const auto& mapping : config.forcing_mappings) {
//             std::cout << "  - Name: " << mapping.name 
//                       << ", File: " << mapping.file 
//                       << std::endl;
//         }

//         std::cout << " -- output -- " << std::endl;
//         std::cout << "Print Interval: " << config.print_interval << std::endl;
//         std::cout << "Output States: " << config.output_states.size() << std::endl;
//         for (const auto& state : config.output_states) {
//             std::cout << "  - State: " << state << std::endl;
//         }
//         std::cout << "Output Path: " << config.output_path << std::endl;
//         std::cout << "Output File: " << config.output_file << std::endl;
//         std::cout << "Final Output: " << (config.final_output) << std::endl;
//         std::cout << "Final Output File: " << config.final_output_file << std::endl;

//         std::cout << " -- solver -- " << std::endl;
//         std::cout << "Solver RTOL: " << config.rtol << std::endl;
//         std::cout << "Solver ATOL: " << config.atol << std::endl;
//         std::cout << "Solver Safety: " << config.safety << std::endl;
//         std::cout << "Solver Min Scale: " << config.min_scale << std::endl;
//         std::cout << "Solver Max Scale: " << config.max_scale << std::endl;
//         std::cout << "Override Tolerances: " << config.override_tolerances << std::endl;
//         std::cout << "Initial Step: " << config.initial_step << std::endl;
//         std::cout << "Override Initial Step: " << config.override_initial_step << std::endl;

//         std::cout << " -- mpi -- " << std::endl;
//         std::cout << "Step Storage: " << config.step_storage << std::endl;
//         std::cout << "Transfer Buffer: " << config.transfer_buffer << std::endl;
//         std::cout << "Discontinuity Buffer: " << config.discontinuity_buf << std::endl;

//         std::cout << " -- flags -- " << std::endl;
//         std::cout << "Uses Dam: " << (config.uses_dam) << std::endl;
//         std::cout << "Convert Area: " << (config.convert_area) << std::endl;
        
//         std::cout << "==============================" << std::endl;
        
//     } catch (const std::exception& e) {
//         std::cerr << "Error: " << e.what() << std::endl;
//         return 1;
//     }
    
//     return 0;
// }