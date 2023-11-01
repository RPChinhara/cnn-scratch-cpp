#include "window.h"

#include <stdexcept>

// Link the necessary libraries
#pragma comment(lib, "d2d1.lib")
#pragma comment(lib, "Gdi32.lib")
#pragma comment(lib, "Ole32.lib")
#pragma comment(lib, "User32.lib")

const char Window::CLASS_NAME[] = "WINDOW";

static RECT agent  = { 5, 985, 55, 1035 }; // Left, Top, Right, Bottom coordinates
static RECT agent2 = { 1850, 985, 1900, 1035 };
static RECT food   = { 5, 5, 55, 55 };
static RECT water  = { 1850, 4, 1900, 50 };

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
        1920,                // Width
        1080,                // Height
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
        case WM_CLOSE: {
            PostQuitMessage(0);
            return 0;
        }
        case WM_KEYDOWN: {
            // Check which key was pressed
            int key = wParam;
            if (key == VK_RIGHT) { // Move right when the right arrow key is pressed
                agent.left += 10; // Move the agent 10 pixels to the right
                agent.right += 10;
                InvalidateRect(hwnd, &agent, TRUE); // Redraw the updated rectangle
            }
            if (key == VK_LEFT) {
                agent.left -= 10;
                agent.right -= 10;
                InvalidateRect(hwnd, &agent, TRUE);
            }
            if (key == VK_UP) {
                agent.top -= 10;
                agent.bottom -= 10;
                InvalidateRect(hwnd, &agent, TRUE);
            }
            if (key == VK_DOWN) {
                agent.top += 10;
                agent.bottom += 10;
                InvalidateRect(hwnd, &agent, TRUE);
            }
            return 0;
        }
        case WM_PAINT: {
            // TODO: Use Direct2D next, and Direct3D 9/10 for 3D?
            PAINTSTRUCT ps;
            HDC hdc = BeginPaint(hwnd, &ps);
            
            // Clear the entire client area with a background color (e.g., white)
            RECT clientRect;
            GetClientRect(hwnd, &clientRect);
            FillRect(hdc, &clientRect, CreateSolidBrush(RGB(255, 255, 255)));
            
            // Draw a rectangle
            FillRect(hdc, &agent, CreateSolidBrush(RGB(0, 0, 0)));
            FillRect(hdc, &agent2, CreateSolidBrush(RGB(0, 0, 0)));
            FillRect(hdc, &food, CreateSolidBrush(RGB(255, 0, 0)));
            FillRect(hdc, &water, CreateSolidBrush(RGB(0, 0, 255)));
            EndPaint(hwnd, &ps);
            return 0;
        }
        default:
            return DefWindowProc(hwnd, uMsg, wParam, lParam);
    }
}