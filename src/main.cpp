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

        value = lower(value);
        value = regex_replace(value, "[\".,!?#$%&()*+/:;<=>@\\[\\]\\^_`{|}~\\\\-]", " ");
        value = regex_replace(value, "\\s*[^\\x00-\\x7f]\\s*", "");
        value = regex_replace(value, "[^\\x00-\\x7f]", "");
        value = regex_replace(value, "'", "");
        value = regex_replace(value, "\\s+", " ");

        data.push_back(value);
    }

    for (auto i = 0; i < 10; ++i)
        std::cout << data[i] << std::endl;

    file.close();

    return tensor();
}

int main() {
    tensor data = load_daily_dialog();

    gru model = gru(0.01f);
    // model.train(x_y_train.first, x_y_train.second);

    return 0;
}