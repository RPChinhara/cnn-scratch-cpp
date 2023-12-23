#pragma once

#include <windows.h>

class Window
{
public:
    Window(HINSTANCE hInst, int nCmdShow);
    int MessageLoop();
    
private:
    static LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

    HWND hwnd;
    HINSTANCE hInstance;
    static const char CLASS_NAME[];
};