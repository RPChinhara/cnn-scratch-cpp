#include "window.h"

#include <stdexcept>

// Link the necessary libraries
#pragma comment(lib, "d2d1.lib")
#pragma comment(lib, "Ole32.lib")
#pragma comment(lib, "User32.lib")

const char Window::CLASS_NAME[] = "WINDOW";

Window::Window(HINSTANCE hInst, int nCmdShow) : hInstance(hInst), hwnd(NULL), pFactory(NULL), pRenderTarget(NULL) {
    // Initialize COM
    CoInitializeEx(NULL, COINIT_APARTMENTTHREADED);

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

    // Initialize Direct2D
    D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, &pFactory);

    // Create a Direct2D render target
    D2D1_RENDER_TARGET_PROPERTIES rtProps = D2D1::RenderTargetProperties(
        D2D1_RENDER_TARGET_TYPE_DEFAULT,
        D2D1::PixelFormat(DXGI_FORMAT_UNKNOWN, D2D1_ALPHA_MODE_PREMULTIPLIED)
    );

    pFactory->CreateHwndRenderTarget(
        rtProps,
        D2D1::HwndRenderTargetProperties(hwnd, D2D1::SizeU(800, 600)),
        &pRenderTarget
    );

    InvalidateRect(hwnd, NULL, FALSE);
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

LRESULT Window::HandleMessage(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    switch (uMsg) {
       case WM_CREATE: {
        Window* pWindow = nullptr;

        CREATESTRUCT* pCreate = reinterpret_cast<CREATESTRUCT*>(lParam);
        pWindow = static_cast<Window*>(pCreate->lpCreateParams);
        SetWindowLongPtr(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(pWindow));
        return pWindow->HandleMessage(hwnd, uMsg, wParam, lParam); // Handle WM_CREATE here
        }
        case WM_PAINT: {
            PAINTSTRUCT ps;
            HDC hdc = BeginPaint(hwnd, &ps);

            // Clear the background
            pRenderTarget->BeginDraw();
            pRenderTarget->Clear(D2D1::ColorF(D2D1::ColorF::White));

            // Draw a red rectangle as a simple 2D character
            ID2D1SolidColorBrush* pBrush = NULL;
            pRenderTarget->CreateSolidColorBrush(D2D1::ColorF(D2D1::ColorF::Blue), &pBrush);

            D2D1_RECT_F rect = D2D1::RectF(100.0f, 100.0f, 200.0f, 200.0f);
            pRenderTarget->FillRectangle(&rect, pBrush);

            pBrush->Release();
            pRenderTarget->EndDraw();

            EndPaint(hwnd, &ps);
            break;
        }
        case WM_CLOSE: {
            DestroyWindow(hwnd);
            break;
        }
        case WM_DESTROY: {
            // Release resources
            if (pRenderTarget != nullptr) {
                pRenderTarget->Release();
                pRenderTarget = nullptr;
            }

            if (pFactory != nullptr) {
                pFactory->Release();
                pFactory = nullptr;
            }

            CoUninitialize();
            PostQuitMessage(0);
            break;
        }
        default: {
            return DefWindowProc(hwnd, uMsg, wParam, lParam);
        }
    }
    return 0;
}

LRESULT CALLBACK Window::WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    // Retrieve the Window instance associated with this hwnd
    Window* pWindow = nullptr;

    if (uMsg == WM_NCCREATE) {
        CREATESTRUCT* pCreate = reinterpret_cast<CREATESTRUCT*>(lParam);
        pWindow = static_cast<Window*>(pCreate->lpCreateParams);
        SetWindowLongPtr(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(pWindow));
    } else {
        // MessageBox(NULL, "The game is over. Please reset the environment.", "Error", MB_ICONERROR | MB_OK);
        pWindow = reinterpret_cast<Window*>(GetWindowLongPtr(hwnd, GWLP_USERDATA));
    }

    if (pWindow) {
        MessageBox(NULL, "The game is over. Please reset the environment.", "Error", MB_ICONERROR | MB_OK);
        return pWindow->HandleMessage(hwnd, uMsg, wParam, lParam);
    } else {
        return DefWindowProc(hwnd, uMsg, wParam, lParam);
    }
}