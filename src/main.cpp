#include "datas.h"
#include "lyrs.h"

int main()
{
    auto data = load_imdb();

    std::cout << data.y[0] << std::endl;
    std::cout << data.y[1] << std::endl;
    std::cout << data.y[2] << std::endl;
    std::cout << data.y[3] << std::endl;
    std::cout << data.y[4] << std::endl;
    std::cout << data.y[5] << std::endl;
    std::cout << data.y[6] << std::endl;
    std::cout << data.y[7] << std::endl;
    std::cout << data.y[8] << std::endl;
    // std::cout << data.x << std::endl;

    std::vector<std::string> vocab = {"foo bar", "bar baz", "baz bada boom"};
    std::vector<std::string> x = {"foo qux bar", "qux baz"};

    const size_t max_tokens = 3;
    const size_t max_len = 2;

    auto vec_y = text_vectorization(vocab, x, max_tokens, max_len);

    std::cout << vec_y << std::endl;

    return 0;
}