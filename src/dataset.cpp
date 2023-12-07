#include "dataset.h"

#include <fstream>
#include <sstream>

Iris LoadIris()
{
    std::ifstream file("datasets\\iris.csv");

    if (!file.is_open()) {
        std::cerr << "Failed to open the file." << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string line;

    std::getline(file, line);

    int idx_features = 0;
    int idx_target   = 0;
    Tensor features = Tensor({ 0.0 }, { 150, 4 });
    Tensor target   = Tensor({ 0.0 }, { 150, 1 });

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        
        std::getline(ss, value, ',');

        std::getline(ss, value, ',');
        features[idx_features] = std::stof(value);
        ++idx_features;
        
        std::getline(ss, value, ',');
        features[idx_features] = std::stof(value);
        ++idx_features;
        
        std::getline(ss, value, ',');
        features[idx_features] = std::stof(value);
        ++idx_features;

        std::getline(ss, value, ',');
        features[idx_features] = std::stof(value);
        ++idx_features;

        std::getline(ss, value);

        if (value == "Iris-setosa") {
            target[idx_target] = 0.0f;
            ++idx_target;
        } else if (value == "Iris-versicolor") {
            target[idx_target] = 1.0f;
            ++idx_target;
        } else if (value == "Iris-virginica") {
            target[idx_target] = 2.0f;
            ++idx_target;
        }
    }
    
    file.close();

    Iris iris;
    iris.features = features;
    iris.target   = target;

    return iris;
}