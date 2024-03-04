#include "dataset.h"
#include "array.h"

#include <fstream>
#include <sstream>
#include <windows.h>

Iris LoadIris()
{
    std::ifstream file("dataset\\iris\\iris.csv");

    if (!file.is_open())
        MessageBox(nullptr, "Failed to open the file", "Error", MB_ICONERROR);

    std::string line;

    std::getline(file, line);

    size_t idx_features = 0;
    size_t idx_target = 0;
    Tensor features = Zeros({150, 4});
    Tensor targets = Zeros({150, 1});

    while (std::getline(file, line))
    {
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

        if (value == "Iris-setosa")
        {
            targets[idx_target] = 0.0f;
            ++idx_target;
        }
        else if (value == "Iris-versicolor")
        {
            targets[idx_target] = 1.0f;
            ++idx_target;
        }
        else if (value == "Iris-virginica")
        {
            targets[idx_target] = 2.0f;
            ++idx_target;
        }
    }

    file.close();

    Iris iris;
    iris.features = features;
    iris.targets = targets;

    return iris;
}