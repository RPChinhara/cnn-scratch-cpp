#include "lyrs.h"

#include <fstream>
#include <sstream>

std::vector<std::string> daily_dialog(const std::string& file_path) {
    std::ifstream file(file_path);

    if (!file.is_open())
        std::cerr << "Failed to open the file: " << file_path << std::endl;

    std::vector<std::string> data;

    std::string line;
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;

        std::getline(ss, value);

        value = lower(value);
        value = regex_replace(value, "[.,!?#$%&()*+/:;<=>@\\[\\]\\^_`{|}~\\\\-]", " ");
        value = regex_replace(value, "\"", "");
        value = regex_replace(value, "\\s*[^\\x00-\\x7f]\\s*", "");
        value = regex_replace(value, "[^\\x00-\\x7f]", "");
        // value = regex_replace(value, "'", "");
        value = regex_replace(value, "\\s+", " ");
        value = regex_replace(value, "\\s+$", "");
        value = value.insert(0, "[START] ");

        data.push_back(value);
    }

    file.close();

    return data;
}

int main() {
    auto input = daily_dialog("datas/daily_dialog/daily_dialog_input.csv");
    auto target = daily_dialog("datas/daily_dialog/daily_dialog_target.csv");

    std::vector<std::string> first_two_input;
    std::vector<std::string> first_two_target;

    for (size_t i = 0; i < 4; ++i)
        first_two_input.push_back(input[i]);

    for (size_t i = 0; i < 4; ++i)
        first_two_target.push_back(target[i]);

    for (size_t i = 0; i < 4; ++i)
        std::cout << first_two_input[i] << std::endl;

    std::cout << text_vectorization(first_two_input, first_two_input, 5000, 4) << std::endl;

     for (size_t i = 0; i < 4; ++i)
        std::cout << first_two_target[i] << std::endl;

    // std::vector<std::string> in = {"say say jim jim jim", "for a say cat how jim", "dog"};

    std::cout << text_vectorization(first_two_input, first_two_target, 5000, 4) << std::endl;

    //TODO: What should max_len be?

    gru model = gru(0.01f);
    // model.train(x_y_train.first, x_y_train.second);

    return 0;
}