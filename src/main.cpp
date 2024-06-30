#include "datas.h"
#include "ten.h"

int main()
{
    ten data = load_aapl();

    std::cout << data << std::endl;

    return 0;
}