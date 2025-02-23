#pragma once

#include <windows.h>

#pragma comment(lib, "user32.lib")
#pragma comment(lib, "gdi32.lib")

class window {
public:
    window(HINSTANCE hInstance);
    ~window();
    bool process_messages();

private:
    static LRESULT CALLBACK window_proc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
    HWND hwnd;
    HINSTANCE hInstance;
};