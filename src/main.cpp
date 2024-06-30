#include "datas.h"
#include "preproc.h"
#include "ten.h"

int main()
{
    const float test_size = 0.2f;

    ten data = load_aapl();
    ten scaled_data = min_max_scaler(data);
    auto train_test = split_dataset(scaled_data, test_size);

    // std::cout << train_test.first << std::endl;
    std::cout << train_test.second << std::endl;

    return 0;
}