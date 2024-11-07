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
    auto input_target = daily_dialog("datas/daily_dialog/daily_dialog.csv");
    auto input = daily_dialog("datas/daily_dialog/daily_dialog_input.csv");
    auto target = daily_dialog("datas/daily_dialog/daily_dialog_target.csv");

    auto input_token = text_vectorization(input_target, input, 5000, 25);
    auto target_token = text_vectorization(input_target, target, 5000, 25);

    gru model = gru(0.01f);
    model.train(input_token, target_token);

    return 0;
}