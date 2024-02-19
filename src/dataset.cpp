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
    Tensor target = Zeros({150, 1});

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
            target[idx_target] = 0.0f;
            ++idx_target;
        }
        else if (value == "Iris-versicolor")
        {
            target[idx_target] = 1.0f;
            ++idx_target;
        }
        else if (value == "Iris-virginica")
        {
            target[idx_target] = 2.0f;
            ++idx_target;
        }
    }

    file.close();

    Iris iris;
    iris.features = features;
    iris.target = target;

    return iris;
}

Tensor ReadMNISTImages(const std::string &filePath)
{
    std::ifstream file(filePath, std::ios::binary);

    if (!file.is_open())
        MessageBox(nullptr, "Failed to open the file", "Error", MB_ICONERROR);

    uint32_t magicNumber, numImages, numRows, numCols;

    // Read the header information
    file.read(reinterpret_cast<char *>(&magicNumber), sizeof(magicNumber));
    file.read(reinterpret_cast<char *>(&numImages), sizeof(numImages));
    file.read(reinterpret_cast<char *>(&numRows), sizeof(numRows));
    file.read(reinterpret_cast<char *>(&numCols), sizeof(numCols));

    // Convert from big-endian to little-endian if needed
    magicNumber = _byteswap_ulong(magicNumber);
    numImages = _byteswap_ulong(numImages);
    numRows = _byteswap_ulong(numRows);
    numCols = _byteswap_ulong(numCols);

    std::vector<std::vector<uint8_t>> images(numImages, std::vector<uint8_t>(numRows * numCols));

    // Read the image data
    for (uint32_t i = 0; i < numImages; ++i)
    {
        file.read(reinterpret_cast<char *>(images[i].data()), numRows * numCols);
    }

    // TODO: I don't like this name
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

    // TODO: Do I need this magicNumber?
    uint32_t magicNumber, numLabels;

    // Read the header information
    file.read(reinterpret_cast<char *>(&magicNumber), sizeof(magicNumber));
    file.read(reinterpret_cast<char *>(&numLabels), sizeof(numLabels));

    // Convert from big-endian to little-endian if needed
    magicNumber = _byteswap_ulong(magicNumber);
    numLabels = _byteswap_ulong(numLabels);

    std::vector<uint8_t> labels(numLabels);

    // Read the label data
    file.read(reinterpret_cast<char *>(labels.data()), numLabels);

    // TODO: I don't like this name
    // TODO: I might need to support 1-dimentional array like (60000,) which is the shape I get when loading MNIST from
    // TensorFlow, but labels for Iris was 2D so I'm not sure.
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
    mnist.trainImages = ReadMNISTImages("dataset\\mnist\\train-images-idx3-ubyte");
    mnist.trainLabels = ReadMNISTLabels("dataset\\mnist\\train-labels-idx1-ubyte");
    mnist.testImages = ReadMNISTImages("dataset\\mnist\\t10k-images-idx3-ubyte");
    mnist.testLabels = ReadMNISTLabels("dataset\\mnist\\t10k-labels-idx1-ubyte");

    return mnist;
}