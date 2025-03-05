#include "window.h"

#pragma comment(lib, "user32.lib")

window::window(HINSTANCE hInstance) : hInstance(hInstance), hwnd(nullptr) {
    const char CLASS_NAME[] = "MyWindowClass";

    WNDCLASSEX wc = {};
    wc.cbSize = sizeof(WNDCLASSEX);
    wc.lpfnWndProc = window_proc;
    wc.hInstance = hInstance;
    wc.lpszClassName = CLASS_NAME;
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);

    RegisterClassEx(&wc);

    hwnd = CreateWindowEx(0, CLASS_NAME, "", WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, 800, 600, NULL, NULL, hInstance, this);

    if (hwnd) {
        ShowWindow(hwnd, SW_SHOW);
        UpdateWindow(hwnd);
        GetCursorPos(&last_mouse_pos);  // sync initial mouse position
    }
}

window::~window() {
    if (hwnd) {
        DestroyWindow(hwnd);
    }
}

LRESULT CALLBACK window::window_proc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    window* win = nullptr;

    if (uMsg == WM_CREATE) {
        CREATESTRUCT* cs = reinterpret_cast<CREATESTRUCT*>(lParam);
        win = static_cast<window*>(cs->lpCreateParams);
        SetWindowLongPtr(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(win));
    } else {
        win = reinterpret_cast<window*>(GetWindowLongPtr(hwnd, GWLP_USERDATA));
    }

    if (win) {
        win->handle_input_message(uMsg, wParam, lParam);
    }

    switch (uMsg) {
        case WM_CLOSE:
            PostQuitMessage(0);
            return 0;
        case WM_DESTROY:
            return 0;
        default:
            return DefWindowProc(hwnd, uMsg, wParam, lParam);
    }
}

bool window::process_messages() {
    MSG msg;
    while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
        if (msg.message == WM_QUIT) return false;
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
    return true;
}

HWND window::get_hwnd() {
    return hwnd;
}

void window::handle_input_message(UINT uMsg, WPARAM wParam, LPARAM lParam) {
    switch (uMsg) {
        case WM_KEYDOWN:
            if (wParam < 256) keys[wParam] = true;
            break;
        case WM_KEYUP:
            if (wParam < 256) keys[wParam] = false;
            break;
        case WM_MOUSEMOVE:
            // You could leave this empty if you call handle_mouse every frame instead
            break;
    }
}

void window::update_camera(camera& cam) {
    const float speed = 0.1f;
    if (keys['A']) cam.move(speed, 0.0f);
    if (keys['D']) cam.move(-speed, 0.0f);
    if (keys['W']) cam.move(0.0f, speed);
    if (keys['S']) cam.move(0.0f, -speed);
}

void window::handle_mouse(camera& cam) {
    POINT current_mouse_pos;
    GetCursorPos(&current_mouse_pos);

    float delta_x = (current_mouse_pos.x - last_mouse_pos.x) * 0.002f;
    float delta_y = (current_mouse_pos.y - last_mouse_pos.y) * 0.002f;

    cam.rotate(delta_x, -delta_y);  // invert Y

    last_mouse_pos = current_mouse_pos;  // update for next frame
}