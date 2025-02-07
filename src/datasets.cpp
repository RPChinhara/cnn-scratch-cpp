#include "datasets.h"
#include "arrs.h"
#include "lyrs.h"
#include "strings.h"

#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdio.h>
#include <vector>

tensor load_aapl() {
    std::ifstream file("datasets/aapl.csv");

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

std::vector<std::string> split_text(const std::string& text, const std::string& delimiter) {
    std::vector<std::string> result;
    size_t pos = 0;
    std::string token;
    std::string s = text;
    while ((pos = s.find(delimiter)) != std::string::npos) {
        token = s.substr(0, pos);
        result.push_back(token);
        s.erase(0, pos + delimiter.length());
    }
    result.push_back(s); // To get the last chunk
    return result;
}

std::pair<std::vector<std::string>, std::vector<std::string>> load_daily_dialog(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file) return {};  // Handle file open failure.

    std::vector<std::string> sources;
    std::vector<std::string> targets;
    std::string line;

    std::getline(file, line);  // Skip header

    static const std::regex special_chars(R"([#$%&()*+/:;<=>@\[\]\^_`{|}~\\-])"); // Keep ?, !, , and .
    static const std::regex quotation_mark(R"(")");
    static const std::regex hyphen(R"(\s-\s)"); // Removes hyphens only between spaces
    static const std::regex non_ascii(R"([^ -~])"); // Faster ASCII check

    while (std::getline(file, line)) {
        // Apply regex transformations
        line = std::regex_replace(line, special_chars, " ");
        line = std::regex_replace(line, quotation_mark, "");
        line = std::regex_replace(line, hyphen, " ");
        line = std::regex_replace(line, non_ascii, "");

        // Split the line by 3 spaces to simulate turns
        std::vector<std::string> turns = split_text(line, "   ");

        // Preprocess each turn
        for (size_t i = 0; i < turns.size(); ++i) {
            std::string src = lower(turns[i]);

            src.erase(src.find_last_not_of(' ') + 1);  // Trim trailing spaces

            // Set tgt to the next turn if available, with SOS and EOS added
            std::string tgt = (i + 1 < turns.size()) ? lower(turns[i + 1]) : "";
            tgt.erase(tgt.find_last_not_of(' ') + 1);  // Trim trailing spaces
            tgt = "<SOS> " + tgt + " <EOS>";

            // Store in separate vectors
            sources.push_back(std::move(src));
            targets.push_back(std::move(tgt));
        }
    }

    return {std::move(sources), std::move(targets)};
}

imdb load_imdb() {
    std::ifstream file("datasets/imdb.csv");

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
    std::ifstream file("datasets/iris.csv");

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

tensor read_mnist_imgs(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);

    uint32_t magic_num, num_imgs, num_rows, num_cols;

    file.read(reinterpret_cast<char*>(&magic_num), sizeof(magic_num));
    file.read(reinterpret_cast<char*>(&num_imgs), sizeof(num_imgs));
    file.read(reinterpret_cast<char*>(&num_rows), sizeof(num_rows));
    file.read(reinterpret_cast<char*>(&num_cols), sizeof(num_cols));

    magic_num = _byteswap_ulong(magic_num);
    num_imgs = _byteswap_ulong(num_imgs);
    num_rows = _byteswap_ulong(num_rows);
    num_cols = _byteswap_ulong(num_cols);

    std::vector<std::vector<uint8_t>> imgs(num_imgs, std::vector<uint8_t>(num_rows * num_cols));

    for (auto i = 0; i < num_imgs; ++i)
        file.read(reinterpret_cast<char*>(imgs[i].data()), num_rows * num_cols);

    tensor imgs_t = zeros({num_imgs, num_rows, num_cols});
    size_t idx = 0;

    for (auto i = 0; i < num_imgs; ++i) {
        for (auto j = 0; j < num_rows; ++j) {
            for (auto k = 0; k < num_cols; ++k) {
                imgs_t[idx] = static_cast<float>(imgs[i][j * num_cols + k]);
                ++idx;
            }
        }
    }

    file.close();

    return imgs_t;
}

tensor read_mnist_labels(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);

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
    data.train_imgs = read_mnist_imgs("datasets/mnist/train-images-idx3-ubyte");
    data.train_labels = read_mnist_labels("datasets/mnist/train-labels-idx1-ubyte");
    data.test_imgs = read_mnist_imgs("datasets/mnist/t10k-images-idx3-ubyte");
    data.test_labels = read_mnist_labels("datasets/mnist/t10k-labels-idx1-ubyte");

    return data;
}