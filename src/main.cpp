#include "window.h"

#include <stdio.h>

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{
    AllocConsole();

    FILE* file;
    freopen_s(&file, "CONOUT$", "w", stdout);

    Window window(hInstance, nCmdShow);
    int result = window.MessageLoop();
    
    FreeConsole();
    fclose(file);

    return result;
}