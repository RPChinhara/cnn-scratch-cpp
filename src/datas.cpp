#include "datas.h"
#include "arrs.h"
#include "preproc.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdio.h>

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

    size_t idx = 0;
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

        std::string text = line.substr(startPos, endPos - startPos);
        std::string text_no_link = regex_replace(text, R"((https?:\/\/|www\.)\S+)", "");
        std::string text_no_html = regex_replace(text_no_link, "<[^>]*>", " ");
        std::string text_sp_around_punc = regex_replace(text_no_html, "([.,!?-])", " $1 ");
        std::string text_no_consecutive_sp = regex_replace(text_sp_around_punc, "\\s{2,}", " ");
        std::string text_no_punc = regex_replace(text_no_consecutive_sp, "[\"#$%&'()*+/:;<=>@\\[\\\\\\]^_`{|}~]", " ");
        std::string text_no_num = regex_replace(text_no_punc, "\\d+", "");
        std::string text_no_ascii = regex_replace(text_no_num, "[^\\x00-\\x7f]", " ");
        std::string text_no_white_sp = regex_replace(text_no_ascii, "\\s+", " ");
        std::string text_no_emoji = regex_replace(text_no_white_sp, "[\xE2\x98\x80-\xE2\x9B\xBF]", "");
        std::string text_spell_corrected = regex_replace(text_no_emoji, "(.)\\1+", "$1$1");

        auto tokens = tokenizer(text_spell_corrected);

        std::cout << "Text: " << idx + 1 << std::endl;
        std::cout << "++++++++++++++++++++++++++: " << std::endl;
        for (auto i = 0; i < tokens.size(); ++i)
        {
            std::cout << tokens[i] << std::endl;
        }
        std::cout << "--------------------------: " << sentiments[idx] << std::endl << std::endl;
        ++idx;
    }

    return imdb();
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