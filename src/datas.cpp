#include "datas.h"
#include "arrs.h"
#include "lyrs.h"
#include "preproc.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <vector>

tensor load_aapl() {
    std::ifstream file("datas/aapl.csv");

    if (!file.is_open())
        std::cerr << "Failed to open the file." << std::endl;

    size_t idx = 0;
    size_t num_datas = 10409;
    size_t num_columns = 1;

    tensor data = zeros({num_datas, num_columns});

    std::string line;
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;

        std::getline(ss, value, ',');
        std::getline(ss, value, ',');
        std::getline(ss, value, ',');
        std::getline(ss, value, ',');
        std::getline(ss, value, ',');

        data[idx] = std::stof(value);
        ++idx;
    }

    file.close();

    return data;
}

imdb load_imdb() {
    std::ifstream file("datas/imdb.csv");

    if (!file.is_open())
        std::cerr << "Failed to open the file." << std::endl;

    std::cout << "Loading imdb dataset..." << std::endl;

    std::vector<std::string> reviews;
    std::vector<float> sentiments;

    std::string line;
    std::getline(file, line);

    while (std::getline(file, line)) {
        size_t end_pos;
        size_t end_pos_positive = line.find(",positive");
        size_t end_pos_negative = line.find(",negative");

        if (end_pos_positive != std::string::npos) {
            end_pos = end_pos_positive;
            sentiments.push_back(1.0f);
        } else if (end_pos_negative != std::string::npos) {
            end_pos = end_pos_negative;
            sentiments.push_back(0.0f);
        }

        std::string text = line.substr(0, end_pos - 0);
        reviews.push_back(text);
    }

    file.close();

    for (auto i = 0; i < reviews.size(); ++i) {
        reviews[i] = lower(reviews[i]);
        reviews[i] = regex_replace(reviews[i], R"((https?:\/\/|www\.)\S+)", "");
        reviews[i] = regex_replace(reviews[i], "<[^>]*>", " ");
        reviews[i] = regex_replace(reviews[i], "\"", "");
        reviews[i] = regex_replace(reviews[i], "[\".,!?#$%&()*+/:;<=>@\\[\\]\\^_`{|}~\\\\-]", " ");
        reviews[i] = regex_replace(reviews[i], "[^\\x00-\\x7f]", " ");
        reviews[i] = regex_replace(reviews[i], "[\xE2\x98\x80-\xE2\x9B\xBF]", "");
        reviews[i] = regex_replace(reviews[i], "\\s+", " ");
        reviews[i] = reviews[i].insert(0, "[START] ");

        auto end_pos = std::find_if(reviews[i].rbegin(), reviews[i].rend(), [](char ch) { return !std::isspace(ch); }).base();
        reviews[i].erase(end_pos, reviews[i].end());
    }

    size_t num_train = std::min(reviews.size(), static_cast<size_t>(25000));
    std::vector<std::string> train(reviews.begin(), reviews.begin() + num_train);

    const size_t max_tokens = 10000;
    const size_t max_len = 200;

    imdb data;
    data.x = text_vectorization(train, reviews, max_tokens, max_len);
    data.y = zeros({sentiments.size(), 1});

    for (auto i = 0; i < sentiments.size(); ++i)
        data.y[i] = sentiments[i];

    return data;
}

iris load_iris() {
    std::ifstream file("datas/iris.csv");

    if (!file.is_open())
        std::cerr << "Failed to open the file." << std::endl;

    size_t idx_x = 0;
    size_t idx_y = 0;

    iris data;
    data.x = zeros({150, 4});
    data.y = zeros({150, 1});

    std::string line;
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;

        std::getline(ss, value, ',');

        std::getline(ss, value, ',');
        data.x[idx_x] = std::stof(value);
        ++idx_x;

        std::getline(ss, value, ',');
        data.x[idx_x] = std::stof(value);
        ++idx_x;

        std::getline(ss, value, ',');
        data.x[idx_x] = std::stof(value);
        ++idx_x;

        std::getline(ss, value, ',');
        data.x[idx_x] = std::stof(value);
        ++idx_x;

        std::getline(ss, value);

        if (value == "Iris-setosa") {
            data.y[idx_y] = 0.0f;
            ++idx_y;
        } else if (value == "Iris-versicolor") {
            data.y[idx_y] = 1.0f;
            ++idx_y;
        } else if (value == "Iris-virginica") {
            data.y[idx_y] = 2.0f;
            ++idx_y;
        }
    }

    file.close();

    return data;
}

tensor read_mnist_imgs(const std::string& filePath) {
    std::ifstream file(filePath, std::ios::binary);

    if (!file.is_open())
        std::cerr << "Failed to open the file." << std::endl;

    uint32_t magic_num, numImages, numRows, numCols;

    file.read(reinterpret_cast<char*>(&magic_num), sizeof(magic_num));
    file.read(reinterpret_cast<char*>(&numImages), sizeof(numImages));
    file.read(reinterpret_cast<char*>(&numRows), sizeof(numRows));
    file.read(reinterpret_cast<char*>(&numCols), sizeof(numCols));

    magic_num = _byteswap_ulong(magic_num);
    numImages = _byteswap_ulong(numImages);
    numRows = _byteswap_ulong(numRows);
    numCols = _byteswap_ulong(numCols);

    std::vector<std::vector<uint8_t>> images(numImages, std::vector<uint8_t>(numRows * numCols));

    for (auto i = 0; i < numImages; ++i)
        file.read(reinterpret_cast<char*>(images[i].data()), numRows * numCols);

    tensor images2 = zeros({numImages, numRows, numCols});
    size_t idx = 0;

    for (auto i = 0; i < numImages; ++i) {
        for (auto j = 0; j < numRows; ++j) {
            for (auto k = 0; k < numCols; ++k) {
                images2[idx] = static_cast<float>(images[i][j * numCols + k]);
                ++idx;
            }
        }
    }

    file.close();

    return images2;
}

tensor read_mnist_labels(const std::string& filePath) {
    std::ifstream file(filePath, std::ios::binary);

    if (!file.is_open())
        std::cerr << "Failed to open the file." << std::endl;

    uint32_t magic_num, num_labels;

    file.read(reinterpret_cast<char*>(&magic_num), sizeof(magic_num));
    file.read(reinterpret_cast<char*>(&num_labels), sizeof(num_labels));

    magic_num = _byteswap_ulong(magic_num);
    num_labels = _byteswap_ulong(num_labels);

    std::vector<uint8_t> labels(num_labels);

    file.read(reinterpret_cast<char*>(labels.data()), num_labels);

    tensor labels2 = zeros({num_labels, 1});
    size_t idx = 0;

    for (auto i = 0; i < num_labels; ++i) {
        labels2[idx] = static_cast<float>(labels[i]);
        ++idx;
    }

    file.close();

    return labels2;
}

mnist load_mnist() {
    mnist data;
    data.train_imgs = read_mnist_imgs("datas/mnist/train-images-idx3-ubyte");
    data.train_labels = read_mnist_labels("datas/mnist/train-labels-idx1-ubyte");
    data.test_imgs = read_mnist_imgs("datas/mnist/t10k-images-idx3-ubyte");
    data.test_labels = read_mnist_labels("datas/mnist/t10k-labels-idx1-ubyte");

    return data;
}