#pragma once

#include <d2d1.h>
#include <windows.h>

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
