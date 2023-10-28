#include "window.h"

#include <stdexcept>

// Link the necessary libraries
#pragma comment(lib, "d2d1.lib")
#pragma comment(lib, "Ole32.lib")
#pragma comment(lib, "User32.lib")

const char Window::CLASS_NAME[] = "WINDOW";

Window::Window(HINSTANCE hInst, int nCmdShow) : hInstance(hInst), hwnd(NULL) {
    // Create a window class
    WNDCLASS wc = {};
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = CLASS_NAME;

    RegisterClass(&wc);

    // Create a window
    hwnd = CreateWindowEx(
        0,                   // Optional window styles
        CLASS_NAME,          // Window class name
        "",                  // Window title
        WS_OVERLAPPEDWINDOW, // Window style
        CW_USEDEFAULT,       // X position
        CW_USEDEFAULT,       // Y position
        800,                 // Width
        600,                 // Height
        NULL,                // Parent window
        NULL,                // Menu
        hInstance,           // Instance handle
        NULL                 // Additional application data
    );

    if (hwnd == NULL) {
        throw std::runtime_error("Failed to create window");
    }

    ShowWindow(hwnd, nCmdShow);
    UpdateWindow(hwnd);
}

int Window::messageLoop() {
    // Main message loop
    MSG msg = {};
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    return static_cast<int>(msg.wParam);
}

LRESULT CALLBACK Window::WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    switch (uMsg) {
        case WM_CLOSE:
            PostQuitMessage(0);
            return 0;
        case WM_PAINT: {
            PAINTSTRUCT ps;
            HDC hdc = BeginPaint(hwnd, &ps);
            // Draw a rectangle
            RECT rect = { 50, 50, 100, 100 }; // Left, Top, Right, Bottom coordinates
            FillRect(hdc, &rect, (HBRUSH)(COLOR_WINDOW + 2)); // Use the default window background color
            EndPaint(hwnd, &ps);
            return 0;
        }
        default:
            return DefWindowProc(hwnd, uMsg, wParam, lParam);
    }
}