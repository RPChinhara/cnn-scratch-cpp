#pragma once
#include <windows.h>
#include "camera.h"

class window {
public:
    window(HINSTANCE hInstance);
    ~window();

    bool process_messages();
    HWND get_hwnd();

    void update_camera(camera& cam);  // keyboard handling
    void handle_mouse(camera& cam);   // mouse handling

private:
    static LRESULT CALLBACK window_proc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
    static window* get_window_from_hwnd(HWND hwnd);

    void handle_input_message(UINT uMsg, WPARAM wParam, LPARAM lParam);

    HINSTANCE hInstance;
    HWND hwnd;

    POINT last_mouse_pos{};
    bool keys[256]{};
};