#pragma once

#include <mutex>
#include <windows.h>

class Window
{
public:
    Window(HINSTANCE hInst, int nCmdShow);
    int MessageLoop();
private:
    HWND hwnd;
    HINSTANCE hInstance;
    static const char CLASS_NAME[];
    static size_t window_width;
    static size_t window_height;
    static std::mutex agentMutex;
    static LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
};