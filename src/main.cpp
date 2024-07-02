#include "arrs.h"
#include "datas.h"
#include "preproc.h"

std::pair<ten, ten> create_sequences(const ten &data, const size_t seq_length)
{
    ten x = zeros({data.size - seq_length - 1, seq_length, 1});
    ten y = zeros({data.size - seq_length - 1, 1});

    size_t idx = 0;
    for (auto i = 0; i < (data.size - seq_length - 1) * seq_length; ++i)
    {
        if (i % seq_length == 0 && i != 0)
            idx -= seq_length - 1;
        x[i] = data[idx];
        ++idx;
    }

    for (auto i = 0; i < data.size - seq_length - 1; ++i)
        y[i] = data[i + seq_length];

    return std::make_pair(x, y);
}

int main()
{
    const float test_size = 0.2f;
    const size_t seq_length = 10;

    ten data = load_aapl();
    ten scaled_data = min_max_scaler(data);
    auto train_test = split(scaled_data, test_size);

    auto x_y_train = create_sequences(train_test.first, seq_length);
    auto x_y_test = create_sequences(train_test.second, seq_length);

    return 0;
}