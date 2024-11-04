#include "lyrs.h"

#include <fstream>
#include <sstream>

tensor daily_dialog(const std::string& file_path) {
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

        value = regex_replace(value, "[.,!?#$%&()*+/:;<=>@\\[\\]\\^_`{|}~\\\\-]", " ");
        value = lower(value);
        value = regex_replace(value, "\"", "");
        value = regex_replace(value, "\\s*[^\\x00-\\x7f]\\s*", "");
        value = regex_replace(value, "[^\\x00-\\x7f]", "");
        // value = regex_replace(value, "'", "");
        value = regex_replace(value, "\\s+", " ");
        value = value.insert(0, "[START] ");

        data.push_back(value);
    }

    file.close();

    for (size_t i = 0; i < 10; ++i) {
        std::cout << data[i] << std::endl;
    }

    return tensor();
}

int main() {
    tensor input = daily_dialog("datas/daily_dialog/daily_dialog_input.csv");
    tensor target = daily_dialog("datas/daily_dialog/daily_dialog_target.csv");

    gru model = gru(0.01f);
    // model.train(x_y_train.first, x_y_train.second);

    std::vector<std::string> vocab = {"[START]", "say", "jim", "how", "about", "going", "for", "a", "few", "beers", "after", "dinner"};
    std::vector<std::string> vocab2 = {"[START] say jim how about going for a few beers after dinner"};
    std::vector<std::string> in = {"say say jim jim jim", "for a say cat how jim", "dog"};

    std::cout << text_vectorization(vocab, in, 5000, 4) << std::endl;
    std::cout << text_vectorization(vocab2, in, 5000, 4) << std::endl;

    return 0;
}