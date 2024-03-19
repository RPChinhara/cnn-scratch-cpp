#include "datasets.h"
#include "arrays.h"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <windows.h>

// std::string ExtractSentence(const std::string &line)
// {
//     size_t startPos = 0;
//     size_t endPosPositive = line.find(",positive");
//     size_t endPosNegative = line.find(",negative");

//     size_t endPos = endPosPositive != std::string::npos ? endPosPositive : endPosNegative;

//     std::string sentence = line.substr(startPos, endPos - startPos);

//     return sentence;
// }

IMDB LoadIMDB()
{
    std::ifstream file("datasets\\IMDB Dataset.csv");

    if (!file.is_open())
        MessageBox(nullptr, "Failed to open the file", "Error", MB_ICONERROR);

    std::vector<std::string> reviews;
    std::vector<float> sentiments;

    std::string line;
    std::getline(file, line);

    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string value;

        size_t startPos = 0;
        size_t endPosPositive = line.find(",positive");
        size_t endPosNegative = line.find(",negative");

        size_t endPos;
        if (endPosPositive != std::string::npos)
        {
            endPos = endPosPositive;
            sentiments.push_back(1.0f);
        }
        else if (endPosNegative != std::string::npos)
        {
            endPos = endPosNegative;
            sentiments.push_back(0.0f);
        }

        std::string sentence = line.substr(startPos, endPos - startPos);

        reviews.push_back(sentence);
        std::cout << "--------------------------: " << sentence << std::endl;
    }

    return IMDB();
}

Iris LoadIris()
{
    std::ifstream file("datasets\\iris.csv");

    if (!file.is_open())
        MessageBox(nullptr, "Failed to open the file", "Error", MB_ICONERROR);

    size_t idxFeatures = 0;
    size_t idxTarget = 0;
    Tensor features = Zeros({150, 4});
    Tensor targets = Zeros({150, 1});

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

    Iris iris;
    iris.features = features;
    iris.targets = targets;

    return iris;
}

Tensor ReadMNISTImages(const std::string &filePath)
{
    std::ifstream file(filePath, std::ios::binary);

    if (!file.is_open())
        MessageBox(nullptr, "Failed to open the file", "Error", MB_ICONERROR);

    uint32_t magicNumber, numImages, numRows, numCols;

    file.read(reinterpret_cast<char *>(&magicNumber), sizeof(magicNumber));
    file.read(reinterpret_cast<char *>(&numImages), sizeof(numImages));
    file.read(reinterpret_cast<char *>(&numRows), sizeof(numRows));
    file.read(reinterpret_cast<char *>(&numCols), sizeof(numCols));

    magicNumber = _byteswap_ulong(magicNumber);
    numImages = _byteswap_ulong(numImages);
    numRows = _byteswap_ulong(numRows);
    numCols = _byteswap_ulong(numCols);

    std::vector<std::vector<uint8_t>> images(numImages, std::vector<uint8_t>(numRows * numCols));

    for (uint32_t i = 0; i < numImages; ++i)
    {
        file.read(reinterpret_cast<char *>(images[i].data()), numRows * numCols);
    }

    Tensor images2 = Zeros({numImages, numRows, numCols});
    size_t idx = 0;

    for (uint32_t i = 0; i < numImages; ++i)
    {
        for (uint32_t j = 0; j < numRows; ++j)
        {
            for (uint32_t k = 0; k < numCols; ++k)
            {
                images2[idx] = static_cast<float>(images[i][j * numCols + k]);
                ++idx;
            }
        }
    }

    file.close();

    return images2;
}

Tensor ReadMNISTLabels(const std::string &filePath)
{
    std::ifstream file(filePath, std::ios::binary);

    if (!file.is_open())
        MessageBox(nullptr, "Failed to open the file", "Error", MB_ICONERROR);

    uint32_t magicNumber, numLabels;

    file.read(reinterpret_cast<char *>(&magicNumber), sizeof(magicNumber));
    file.read(reinterpret_cast<char *>(&numLabels), sizeof(numLabels));

    magicNumber = _byteswap_ulong(magicNumber);
    numLabels = _byteswap_ulong(numLabels);

    std::vector<uint8_t> labels(numLabels);

    file.read(reinterpret_cast<char *>(labels.data()), numLabels);

    Tensor labels2 = Zeros({numLabels, 1});
    size_t idx = 0;

    for (uint32_t i = 0; i < numLabels; ++i)
    {
        labels2[idx] = static_cast<float>(labels[i]);
        ++idx;
    }

    file.close();

    return labels2;
}

MNIST LoadMNIST()
{
    MNIST mnist;
    mnist.trainImages = ReadMNISTImages("datasets\\mnist\\train-images-idx3-ubyte");
    mnist.trainLabels = ReadMNISTLabels("datasets\\mnist\\train-labels-idx1-ubyte");
    mnist.testImages = ReadMNISTImages("datasets\\mnist\\t10k-images-idx3-ubyte");
    mnist.testLabels = ReadMNISTLabels("datasets\\mnist\\t10k-labels-idx1-ubyte");

    return mnist;
}