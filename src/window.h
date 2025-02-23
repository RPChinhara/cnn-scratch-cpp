#pragma once

#include <windows.h>

class window {
public:
    window(HINSTANCE hInstance);
    ~window();
    bool process_messages();
    HWND get_hwnd();

private:
    static LRESULT CALLBACK window_proc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
    HWND hwnd;
    HINSTANCE hInstance;
};