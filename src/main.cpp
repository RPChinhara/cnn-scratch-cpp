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