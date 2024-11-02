#include "lyrs.h"

#include <fstream>
#include <sstream>

tensor load_daily_dialog() {
    std::ifstream file("datas/daily_dialog.csv");

    if (!file.is_open())
        std::cerr << "Failed to open the file." << std::endl;

    std::vector<std::string> data;

    std::string line;
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;

        std::getline(ss, value, ']');

        value = regex_replace(value, R"((https?:\/\/|www\.)\S+)", "");
        value = regex_replace(value, "<[^>]*>", " ");
        value = regex_replace(value, "\"", "");
        value = regex_replace(value, "[\".,!?#$%&()*+/:;<=>@\\[\\]\\^_`{|}~\\\\-]", " ");
        value = regex_replace(value, "[^\\x00-\\x7f]", " ");
        value = regex_replace(value, "[\xE2\x98\x80-\xE2\x9B\xBF]", "");
        value = regex_replace(value, "\\s+", " ");

        data.push_back(value);
    }

    std::cout << data[0] << std::endl;
    std::cout << data[1] << std::endl;
    std::cout << data[data.size() - 1] << std::endl;

    file.close();

    return tensor();
}

int main() {
    tensor data = load_daily_dialog();

    gru model = gru(0.01f);
    // model.train(x_y_train.first, x_y_train.second);

    return 0;
}