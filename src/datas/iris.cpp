#include "iris.h"
#include "arrs.h"

#include <fstream>
#include <sstream>

iris load_iris()
{
    std::ifstream file("datas/iris.csv");

    if (!file.is_open())
        std::cerr << "Failed to open the file." << std::endl;

    size_t idxFeatures = 0;
    size_t idxTarget = 0;
    Ten features = zeros({150, 4});
    Ten targets = zeros({150, 1});

    std::string line;
    std::getline(file, line);

    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string value;

        std::getline(ss, value, ',');

        std::getline(ss, value, ',');
        features[idxFeatures] = std::stof(value);
        ++idxFeatures;

        std::getline(ss, value, ',');
        features[idxFeatures] = std::stof(value);
        ++idxFeatures;

        std::getline(ss, value, ',');
        features[idxFeatures] = std::stof(value);
        ++idxFeatures;

        std::getline(ss, value, ',');
        features[idxFeatures] = std::stof(value);
        ++idxFeatures;

        std::getline(ss, value);

        if (value == "Iris-setosa")
        {
            targets[idxTarget] = 0.0f;
            ++idxTarget;
        }
        else if (value == "Iris-versicolor")
        {
            targets[idxTarget] = 1.0f;
            ++idxTarget;
        }
        else if (value == "Iris-virginica")
        {
            targets[idxTarget] = 2.0f;
            ++idxTarget;
        }
    }

    file.close();

    iris data;
    data.features = features;
    data.targets = targets;

    return data;
}