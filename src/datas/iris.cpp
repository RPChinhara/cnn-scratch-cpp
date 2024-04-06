#include "iris.h"
#include "arrs.h"

#include <fstream>
#include <sstream>

iris load_iris()
{
    std::ifstream file("datas/iris.csv");

    if (!file.is_open())
        std::cerr << "Failed to open the file." << std::endl;

    size_t idx_x = 0;
    size_t idx_y = 0;
    
    ten x = zeros({150, 4});
    ten y = zeros({150, 1});

    std::string line;
    std::getline(file, line);

    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string value;

        std::getline(ss, value, ',');

        std::getline(ss, value, ',');
        x[idx_x] = std::stof(value);
        ++idx_x;

        std::getline(ss, value, ',');
        x[idx_x] = std::stof(value);
        ++idx_x;

        std::getline(ss, value, ',');
        x[idx_x] = std::stof(value);
        ++idx_x;

        std::getline(ss, value, ',');
        x[idx_x] = std::stof(value);
        ++idx_x;

        std::getline(ss, value);

        if (value == "Iris-setosa")
        {
            y[idx_y] = 0.0f;
            ++idx_y;
        }
        else if (value == "Iris-versicolor")
        {
            y[idx_y] = 1.0f;
            ++idx_y;
        }
        else if (value == "Iris-virginica")
        {
            y[idx_y] = 2.0f;
            ++idx_y;
        }
    }

    file.close();

    iris data;
    data.x = x;
    data.y = y;

    return data;
}