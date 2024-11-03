#include "lyrs.h"

#include <fstream>
#include <sstream>

tensor daily_dialog_input() {
    std::ifstream file("datas/daily_dialog/daily_dialog_input.csv");

    if (!file.is_open())
        std::cerr << "Failed to open the file." << std::endl;

    std::vector<std::string> input;

    std::string line;
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;

        std::getline(ss, value);

        value = lower(value);
        value = regex_replace(value, "[\".,!?#$%&()*+/:;<=>@\\[\\]\\^_`{|}~\\\\-]", " ");
        value = regex_replace(value, "\\s*[^\\x00-\\x7f]\\s*", "");
        value = regex_replace(value, "[^\\x00-\\x7f]", "");
        // value = regex_replace(value, "'", "");
        value = regex_replace(value, "\\s+", " ");

        input.push_back(value);
    }

    file.close();

    return tensor();
}

tensor daily_dialog_target() {
    std::ifstream file("datas/daily_dialog/daily_dialog_target.csv");

    if (!file.is_open())
        std::cerr << "Failed to open the file." << std::endl;

    std::vector<std::string> target;

    std::string line;
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;

        std::getline(ss, value);

        value = lower(value);
        value = regex_replace(value, "[\".,!?#$%&()*+/:;<=>@\\[\\]\\^_`{|}~\\\\-]", " ");
        value = regex_replace(value, "\\s*[^\\x00-\\x7f]\\s*", "");
        value = regex_replace(value, "[^\\x00-\\x7f]", "");
        // value = regex_replace(value, "'", "");
        value = regex_replace(value, "\\s+", " ");

        target.push_back(value);
    }

    for (auto i = 0; i < 10; ++i)
        std::cout << target[i] << std::endl;

    file.close();

    return tensor();
}

int main() {
    tensor x = daily_dialog_input();
    tensor y = daily_dialog_target();

    gru model = gru(0.01f);
    // model.train(x_y_train.first, x_y_train.second);

    return 0;
}