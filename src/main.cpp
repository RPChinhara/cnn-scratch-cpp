#include "window.h"
#include "random.h"
#include "tensor.h"
#include "linalg.h"
#include "arrays.h"

#include <iostream>

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    AllocConsole();
    freopen_s((FILE**)stdout, "CONOUT$", "w", stdout);

    std::cout << 0 % 3 << std::endl;
    std::cout << 1 % 3 << std::endl;
    std::cout << 2 % 3 << std::endl;
    std::cout << 3 % 3 << std::endl;
    std::cout << 4 % 3 << std::endl;
    std::cout << 5 % 3 << std::endl;

    Tensor a = Ones({ 2, 3 });
    Tensor b = Ones({ 2, 3 });
    Tensor c = Ones({ 1, 3 });
    Tensor d = Tensor({ 1, 2, 3, 4, 5, 6 }, { 2, 3 });
    Tensor e = Tensor({ 1, 2, 3, 4, 5, 6 }, { 2, 3 });
    Tensor f = Tensor({ 1, 2, 3 }, { 1, 3 });
    // Tensor g = Tensor({ 1, 2, 3 }, { 1, 0 });
    Tensor h = Tensor({ 1, 2, 3, 5 }, { 1, 4 });
    Tensor ffdf = normal_distribution( { 2, 3 });
    Tensor df = uniform_distribution( { 2, 3 });

    std::cout << a + b << std::endl;
    std::cout << a + c << std::endl;
    std::cout << d + e << std::endl;
    std::cout << d + f << std::endl;
    std::cout << Transpose(d) << std::endl;
    std::cout << d._num_ch_dim << std::endl;
    std::cout << h._num_ch_dim << std::endl;
    std::cout << ffdf << std::endl;
    std::cout << df << std::endl;

    try {
        Window window(hInstance, nCmdShow);
        int result = window.messageLoop();
        FreeConsole();
        return result;
    } catch (const std::exception& e) {
        MessageBox(nullptr, e.what(), "Error", MB_ICONERROR | MB_OK);
        return -1;
    }
}