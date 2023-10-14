#pragma once

#include <windows.h>

class Window {
public:
    Window(HINSTANCE hInst, int nCmdShow);
    int messageLoop();
    
private:
    static const char CLASS_NAME[];
    HWND hwnd;
    HINSTANCE hInstance;
    static LRESULT CALLBACK WindowProcedure(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
};
