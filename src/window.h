#pragma once

#include <windows.h>

// Link the necessary libraries
#pragma comment(lib, "d2d1.lib")
#pragma comment(lib, "User32.lib")

class Window {
public:
    Window(HINSTANCE hInst, int nCmdShow);
    int messageLoop();
    
private:
    static const char CLASS_NAME[];
    HWND hwnd;
    HINSTANCE hInstance;
    static LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
};
