#include "window.h"
#include "random.h"
#include "tensor.h"
#include "linalg.h"
#include "array.h"
#include "mathematics.h"
#include "linalg.h"

#include <iostream>

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{
    AllocConsole();
    freopen_s((FILE**)stdout, "CONOUT$", "w", stdout);

    // std::cout << 0 % 3 << std::endl;
    // std::cout << 1 % 3 << std::endl;
    // std::cout << 2 % 3 << std::endl;
    // std::cout << 3 % 3 << std::endl;
    // std::cout << 4 % 3 << std::endl;
    // std::cout << 5 % 3 << std::endl;

    Tensor a1 = Ones({ 2, 3, 4, 2 });
    Tensor a = Ones({ 2, 3 });
    Tensor b = Ones({ 3, 2 });
    Tensor c = Ones({ 2, 2, 3 });
    Tensor d = Tensor({ 1, 2, 3, 4, 5, 6 }, { 2, 3 });
    Tensor e = Tensor({ 1, 23, 33, 4, 5, 6 }, { 2, 3 });
    Tensor f = Tensor({ 1, 2, 3 }, { 1, 3 });
    // Tensor g = Tensor({ 1, 2, 3 }, { 1, 0 });
    Tensor h = Tensor({ 1, 2, 3, 5 }, { 1, 4 });
    Tensor ffdf = NormalDistribution( { 2, 3 });
    Tensor df = UniformDistribution( { 2, 3, 3 });

    // std::cout << a + b << std::endl;
    // std::cout << a + c << std::endl;
    // std::cout << d + e << std::endl;
    // std::cout << d + f << std::endl;
    std::cout << Transpose(d) << std::endl;
    // std::cout << d.num_ch_dim << std::endl;
    // std::cout << h.num_ch_dim << std::endl;
    // std::cout << ffdf << std::endl;
    // std::cout << df << std::endl;
    std::cout << MatMul(a, b) << std::endl;
    std::cout << Maximum(d, e) << std::endl;
    std::cout << c << std::endl;

    try {
        Window window(hInstance, nCmdShow);
        int result = window.MessageLoop();
        FreeConsole();
        return result;
    } catch (const std::exception& e) {
        MessageBox(nullptr, e.what(), "Error", MB_ICONERROR | MB_OK);
        return -1;
    }
}