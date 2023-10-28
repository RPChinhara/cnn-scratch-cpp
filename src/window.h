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
     // Direct2D variables
    ID2D1Factory* pFactory;
    ID2D1HwndRenderTarget* pRenderTarget;
    LRESULT HandleMessage(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
    static LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
};
