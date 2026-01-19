#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

std::vector<std::vector<double>> readCSV(const std::string& filename) {
    std::vector<std::vector<double>> data;
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file " + filename);
    }

    std::string line;
    while (std::getline(file, line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string cell;

        while (std::getline(ss, cell, ',')) {
            row.push_back(std::stod(cell));  // convert string → double
        }
        data.push_back(row);
    }

    return data;
}


std::vector<double> flatten(std::vector<std::vector<double>> matrix){
    size_t len = size(matrix);
    std::vector<double> res;
    for (size_t i = 0; i < len; i++){
        res.push_back(matrix[i][0]);
    }
    return res;
}