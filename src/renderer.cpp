#include "renderer.h"

renderer::renderer() {}

renderer::~renderer() {
    shutdown();
}

bool renderer::init() {
    D3D_FEATURE_LEVEL feature_levels[] = {
        D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_1,
        D3D_FEATURE_LEVEL_10_0
    };
    D3D_FEATURE_LEVEL feature_level_out;

    HRESULT hr = D3D11CreateDevice(
        nullptr,
        D3D_DRIVER_TYPE_HARDWARE,
        nullptr,
        0,
        feature_levels,
        _countof(feature_levels),
        D3D11_SDK_VERSION,
        device.GetAddressOf(),
        &feature_level_out,
        device_context.GetAddressOf()
    );

    if (FAILED(hr)) {
        OutputDebugString("Failed to initialize Direct3D 11 device.\n");
        return false;
    }

    OutputDebugString("Direct3D 11 device initialized successfully!\n");
    MessageBox(NULL, "Direct3D 11 device initialized successfully!", "Message", MB_OK);

    return true;
}

void renderer::render() {
}

void renderer::shutdown() {
    // ComPtr automatically releases resources, so no need to manually release.
}
