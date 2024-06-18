#include "datas.h"
#include "arrs.h"
#include "lyrs.h"
#include "preproc.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <vector>

en_es load_en_es()
{
    std::locale::global(std::locale("es_ES.UTF-8"));

    std::wifstream file("datas/spa.txt");

    if (!file)
        std::cerr << "Failed to open the file." << std::endl;

    std::vector<std::wstring> x;
    std::vector<std::wstring> y;

    std::wstring line;
    while (std::getline(file, line))
    {
        size_t tab_pos = line.find(L"\t");

        std::wstring en_part = line.substr(0, tab_pos);
        std::wstring es_part = line.substr(tab_pos + 1);

        x.push_back(es_part);
        y.push_back(en_part);
    }

    file.close();

    en_es data;
    data.x = x;
    data.y = y;

    return data;
}

imdb load_imdb()
{
    std::ifstream file("datas/IMDB Dataset.csv");

    if (!file.is_open())
        std::cerr << "Failed to open the file." << std::endl;

    std::vector<std::string> reviews;
    std::vector<float> sentiments;

    std::string line;
    std::getline(file, line);

    while (std::getline(file, line))
    {
        size_t end_pos;
        size_t end_pos_positive = line.find(",positive");
        size_t end_pos_negative = line.find(",negative");

        if (end_pos_positive != std::string::npos)
        {
            end_pos = end_pos_positive;
            sentiments.push_back(1.0f);
        }
        else if (end_pos_negative != std::string::npos)
        {
            end_pos = end_pos_negative;
            sentiments.push_back(0.0f);
        }

        std::string text = line.substr(0, end_pos - 0);
        reviews.push_back(text);
    }

    file.close();

    for (auto i = 0; i < reviews.size(); ++i)
    {
        reviews[i] = lower(reviews[i]);
        reviews[i] = regex_replace(reviews[i], R"((https?:\/\/|www\.)\S+)", "");
        reviews[i] = regex_replace(reviews[i], "<[^>]*>", " ");
        reviews[i] = regex_replace(reviews[i], "\"", "");
        reviews[i] = regex_replace(reviews[i], "[\".,!?#$%&()*+/:;<=>@\\[\\]\\^_`{|}~\\\\-]", " ");
        reviews[i] = regex_replace(reviews[i], "[^\\x00-\\x7f]", " ");
        reviews[i] = regex_replace(reviews[i], "[\xE2\x98\x80-\xE2\x9B\xBF]", "");
        reviews[i] = regex_replace(reviews[i], "\\s+", " ");

        auto end_pos =
            std::find_if(reviews[i].rbegin(), reviews[i].rend(), [](char ch) { return !std::isspace(ch); }).base();
        reviews[i].erase(end_pos, reviews[i].end());
    }

    size_t num_train = std::min(reviews.size(), static_cast<size_t>(25000));
    std::vector<std::string> train(reviews.begin(), reviews.begin() + num_train);

    const size_t max_tokens = 10000;
    const size_t max_len = 200;

    std::cout << train.back() << std::endl;

    imdb data;
    data.x = text_vectorization(train, reviews, max_tokens, max_len);
    data.y = zeros({sentiments.size(), 1});

    for (auto i = 0; i < sentiments.size(); ++i)
        data.y[i] = sentiments[i];

    return data;
}

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

    for (auto i = 0; i < numImages; ++i)
    {
        file.read(reinterpret_cast<char *>(images[i].data()), numRows * numCols);
    }

    ten images2 = zeros({numImages, numRows, numCols});
    size_t idx = 0;

    for (auto i = 0; i < numImages; ++i)
    {
        for (auto j = 0; j < numRows; ++j)
        {
            for (auto k = 0; k < numCols; ++k)
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

    for (auto i = 0; i < numLabels; ++i)
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