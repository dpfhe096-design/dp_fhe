#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

std::vector<std::vector<double>> readCSV(const std::string& filename);

std::vector<double> flatten(std::vector<std::vector<double>> matrix);