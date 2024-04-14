#include "mnist.h"
#include "arrs.h"

#include <fstream>
#include <string>
#include <vector>

ten ReadMNISTImages(const std::string &filePath)
{
    std::ifstream file(filePath, std::ios::binary);

    if (!file.is_open())
        std::cerr << "Failed to open the file." << std::endl;

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

    ten images2 = zeros({numImages, numRows, numCols});
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

ten ReadMNISTLabels(const std::string &filePath)
{
    std::ifstream file(filePath, std::ios::binary);

    if (!file.is_open())
        std::cerr << "Failed to open the file." << std::endl;

    uint32_t magicNumber, numLabels;

    file.read(reinterpret_cast<char *>(&magicNumber), sizeof(magicNumber));
    file.read(reinterpret_cast<char *>(&numLabels), sizeof(numLabels));

    magicNumber = _byteswap_ulong(magicNumber);
    numLabels = _byteswap_ulong(numLabels);

    std::vector<uint8_t> labels(numLabels);

    file.read(reinterpret_cast<char *>(labels.data()), numLabels);

    ten labels2 = zeros({numLabels, 1});
    size_t idx = 0;

    for (uint32_t i = 0; i < numLabels; ++i)
    {
        labels2[idx] = static_cast<float>(labels[i]);
        ++idx;
    }

    file.close();

    return labels2;
}

mnist load_mnist()
{
    mnist data;
    data.trainImages = ReadMNISTImages("datas/mnist/train-images-idx3-ubyte");
    data.trainLabels = ReadMNISTLabels("datas/mnist/train-labels-idx1-ubyte");
    data.testImages = ReadMNISTImages("datas/mnist/t10k-images-idx3-ubyte");
    data.testLabels = ReadMNISTLabels("datas/mnist/t10k-labels-idx1-ubyte");

    return data;
}