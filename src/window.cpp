#include "window.h"

#include <stdexcept>
#include <iostream>

// Link the necessary libraries
#pragma comment(lib, "d2d1.lib")
#pragma comment(lib, "Gdi32.lib")
#pragma comment(lib, "Ole32.lib")
#pragma comment(lib, "User32.lib")

static int window_width  = 1920;
static int window_height = 1080;
const char Window::CLASS_NAME[] = "WINDOW";

static RECT agent  = { 5, 895, 55, 945 }; // Left, Top, Right, Bottom coordinates
static RECT agent2 = { 1850, 895, 1900, 945 };
static RECT food   = { 5, 5, 55, 55 };
static RECT water  = { 1850, 4, 1900, 50 };

void CheckBoundaryCollision(RECT& rect) {
    // Check and handle collisions with the window boundaries
    if (rect.left < 0) {
        rect.left = 0;
        rect.right = rect.left + (rect.right - rect.left);
    }
    if (rect.top < 0) {
        rect.top = 0;
        rect.bottom = rect.top + (rect.bottom - rect.top);
    }
    if (rect.right > window_width) {
        rect.right = window_width;
        rect.left = rect.right - (rect.right - rect.left);
    }
    if (rect.bottom > window_height) {
        rect.bottom = window_height;
        rect.top = rect.bottom - (rect.bottom - rect.top);
    }
}

bool IsColliding(const RECT& rect1, const RECT& rect2) {
    // Check for collision between two rectangles
    return (rect1.left < rect2.right &&
            rect1.right > rect2.left &&
            rect1.top < rect2.bottom &&
            rect1.bottom > rect2.top);
}

void ResolveCollision(RECT& movingRect, const RECT& staticRect) {
    // Determine the horizontal and vertical distances between rectangles
    int horizontalDistance = std::min(abs(staticRect.right - movingRect.left), abs(movingRect.right - staticRect.left));
    int verticalDistance = std::min(abs(staticRect.bottom - movingRect.top), abs(movingRect.bottom - staticRect.top));

    if (horizontalDistance < 10 || verticalDistance < 10) {
        // If the rectangles are too close, stop the moving rectangle
        return;
    }

    // Move the rectangle to the right
    movingRect.left += 10;
    movingRect.right += 10;
}

Window::Window(HINSTANCE hInst, int nCmdShow) : hInstance(hInst), hwnd(NULL) {
    // Create a window class
    WNDCLASS wc = {};
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = CLASS_NAME;

    if (!RegisterClass(&wc)) {
        MessageBox(NULL, "Window Registration Failed!", "Error", MB_ICONERROR);
    }

    // Create a window
    hwnd = CreateWindow(
        CLASS_NAME,          // Window class name
        "",                  // Window title
        WS_OVERLAPPEDWINDOW, // Window style
        CW_USEDEFAULT,       // X position
        CW_USEDEFAULT,       // Y position
        window_width,        // Width
        window_height,       // Height
        NULL,                // Parent window
        NULL,                // Menu
        hInstance,           // Instance handle
        NULL                 // Additional application data
    );

    if (hwnd == NULL) {
        MessageBox(NULL, "Window Creation Failed!", "Error", MB_ICONERROR);
    }

    ShowWindow(hwnd, nCmdShow);
    UpdateWindow(hwnd);
}

HWND Window::get_hwnd() {
    return hwnd;
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
                
                // Check for collision with the static rectangle
                if (IsColliding(agent, agent2)) {
                    ResolveCollision(agent, agent2); // Resolve the collision by adjusting the position of the moving rectangle
                } else if (IsColliding(agent, water)) {
                    ResolveCollision(agent, water);
                } else {
                    agent.left += 5; // Move the agent 10 pixels to the right
                    agent.right += 5;
                }

                 // Check for boundary collision
                CheckBoundaryCollision(agent);

                InvalidateRect(hwnd, NULL, TRUE); // Redraw the updated rectangle
            }
            if (key == VK_LEFT) {
                agent.left -= 5;
                agent.right -= 5;
                InvalidateRect(hwnd, NULL, TRUE);
            }
            if (key == VK_UP) {
                agent.top -= 5;
                agent.bottom -= 5;
                InvalidateRect(hwnd, NULL, TRUE);
            }
            if (key == VK_DOWN) {
                agent.top += 5;
                agent.bottom += 5;
                InvalidateRect(hwnd, NULL, TRUE);
            }
            return 0;
        }
        case WM_SIZE: {
            // RECT clientRect;
            // GetClientRect(hwnd, &clientRect);
            // // Handle window resizing
            // int clientWidth = LOWORD(lParam);
            // int clientHeight = HIWORD(lParam);

            // // Calculate the position of the agent as a percentage of the client area
            // agentXPercent = 0.005; // 0.5% from the left
            // agentYPercent = 0.995; // 99.5% from the top

            // // Calculate the new positions and sizes of your rectangles here
            // agentWidth = static_cast<int>(clientWidth * 50 / clientWidth);
            // agentHeight = static_cast<int>(clientHeight * 50 / clientHeight);

            // agent2 = {
            //     static_cast<int>(clientRect.right * agentXPercent),
            //     static_cast<int>(clientRect.bottom * agentYPercent) - agentHeight,
            //     static_cast<int>(clientRect.right * agentXPercent) + agentWidth,
            //     static_cast<int>(clientRect.bottom * agentYPercent)
            // };

            // Redraw the scene
            InvalidateRect(hwnd, NULL, TRUE);
            return 0;
        }
        case WM_PAINT: {
            // TODO: Use Direct2D next, and Direct3D 9/10 for 3D?
            PAINTSTRUCT ps;
            HDC hdc = BeginPaint(hwnd, &ps);
            
            // Clear the entire client area with a background color (e.g., white)
            RECT clientRect;
            GetClientRect(hwnd, &clientRect);
            FillRect(hdc, &clientRect, CreateSolidBrush(RGB(34, 139, 34)));

            // Draw a rectangle
            FillRect(hdc, &agent, CreateSolidBrush(RGB(218, 171, 145)));
            FillRect(hdc, &agent2, CreateSolidBrush(RGB(218, 171, 145)));
            FillRect(hdc, &food, CreateSolidBrush(RGB(255, 0, 0)));
            FillRect(hdc, &water, CreateSolidBrush(RGB(0, 0, 255)));

            EndPaint(hwnd, &ps);
            return 0;
        }
        default:
            return DefWindowProc(hwnd, uMsg, wParam, lParam);
    }
}