#include <iostream>

#include "renderer.h"

renderer::renderer(HWND hwnd) : hwnd(hwnd) {}

renderer::~renderer() {
}

bool renderer::init() {
    HRESULT hr = D3D11CreateDevice(
        nullptr,                      // Default adapter
        D3D_DRIVER_TYPE_HARDWARE,     // Use GPU
        nullptr,                      // No software renderer
        0,                            // No special flags
        nullptr, 0,                   // Auto-select feature level
        D3D11_SDK_VERSION,
        device.GetAddressOf(),
        nullptr,                      // Don't need feature level output
        device_context.GetAddressOf()
    );

    if (FAILED(hr)) {
        std::cerr << "Failed to initialize Direct3D 11 device.\n";
        return false;
    }

    std::cout << "Direct3D 11 device initialized successfully!\n";

    return true;
}

void renderer::render() {
}