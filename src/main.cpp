#include "datas.h"
#include "preproc.h"
#include "ten.h"

int main()
{
    ten data = load_aapl();
    ten scaled_data = min_max_scaler(data);

    std::cout << scaled_data << std::endl;

    return 0;
}