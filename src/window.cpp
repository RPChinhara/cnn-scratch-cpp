#include "window.h"
#include "audio_player.h"
#include "physics.h"

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

static RECT agent  = { 7, 895, 57, 945 }; // Left, Top, Right, Bottom coordinates
static RECT agent2 = { 1850, 895, 1900, 945 };
static RECT food   = { 5, 5, 55, 55 };
static RECT water  = { 1850, 4, 1900, 50 };
static RECT bed    = { 5, 865, 60, 965 };

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

int Window::messageLoop() {
    // Main message loop
    MSG msg = {};
    AudioPlayer soundPlayer(hwnd);

    if (!soundPlayer.Initialize()) {
        // Handle initialization error
        MessageBox(NULL, "1", "Error", MB_ICONERROR);
        return -1;
    }

    if (!soundPlayer.LoadAudioData("assets\\mixkit-city-traffic-background-ambience-2930.wav")) {
        // Handle audio data loading error
        MessageBox(NULL, "2", "Error", MB_ICONERROR);
        return -1;
    }
    
    // TODO: No sound playing
    // Play the sound
    if (!soundPlayer.PlaySound()) {
        // Handle sound playback error
        MessageBox(NULL, "Failed to play sound!", "Error", MB_ICONERROR);
        return -1;
    }

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
                CheckBoundaryCollision(agent, window_width, window_height);

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
            FillRect(hdc, &bed, CreateSolidBrush(RGB(255, 255, 255)));
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