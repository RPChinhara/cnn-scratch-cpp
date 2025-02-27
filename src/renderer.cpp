#include <iostream>

#include "renderer.h"

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")

renderer::renderer(HWND hwnd) : hwnd(hwnd) {}

renderer::~renderer() {
    cleanup();
}

bool renderer::create_device_and_swap_chain() {
    DXGI_SWAP_CHAIN_DESC sc_desc = {};
    sc_desc.BufferDesc.Width = 800;
    sc_desc.BufferDesc.Height = 600;
    sc_desc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sc_desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sc_desc.BufferCount = 1;
    sc_desc.OutputWindow = hwnd;
    sc_desc.Windowed = TRUE;
    sc_desc.SampleDesc.Count = 1;
    sc_desc.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;

    HRESULT hr = D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &sc_desc, swap_chain.GetAddressOf(), device.GetAddressOf(), nullptr, device_context.GetAddressOf());

    if (FAILED(hr)) {
        std::cerr << "Failed to create Direct3D 11 device and swap.\n";
        return false;
    }

    return true;
}

bool renderer::create_render_target() {
    Microsoft::WRL::ComPtr<ID3D11Texture2D> back_buffer; // The off-screen buffer where DirectX draws the next frame
    HRESULT hr = swap_chain->GetBuffer(0, __uuidof(ID3D11Texture2D), (void**)back_buffer.GetAddressOf());
    if (FAILED(hr)) {
        std::cerr << "Failed to get back buffer.\n";
        return false;
    }

    hr = device->CreateRenderTargetView(back_buffer.Get(), nullptr, render_target.GetAddressOf());
    if (FAILED(hr)) {
        std::cerr << "Failed to create render target.\n";
        return false;
    }

    device_context->OMSetRenderTargets(1, render_target.GetAddressOf(), nullptr);
    return true;
}

void renderer::create_viewport(float window_width, float window_height) {
    // NOTE:  Defines the area where DirectX will draw graphics inside the window because by default, DirectX does not know where to draw. We need to tell it how large the rendering area is.
    D3D11_VIEWPORT viewport = {};
    viewport.Width = window_width;
    viewport.Height = window_height;
    viewport.MinDepth = 0.0f;  // Closest depth (near plane)
    viewport.MaxDepth = 1.0f;  // Farthest depth (far plane)
    viewport.TopLeftX = 0;
    viewport.TopLeftY = 0;

    device_context->RSSetViewports(1, &viewport);
}

void renderer::create_depth_buffer(int width, int height) {
    // NOTE: Ensures correct depth sorting so that closer objects appear in front of farther objects. Without it, objects might overlap incorrectly, ignoring their depth. Essential for 3D rendering (not needed for 2D).

    // 1️⃣ Create a depth buffer texture - a texture that stores depth values
    D3D11_TEXTURE2D_DESC depth_desc = {};
    depth_desc.Width = width;
    depth_desc.Height = height;
    depth_desc.MipLevels = 1;
    depth_desc.ArraySize = 1;
    depth_desc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;  // 24-bit depth, 8-bit stencil
    depth_desc.SampleDesc.Count = 1;  // No MSAA
    depth_desc.Usage = D3D11_USAGE_DEFAULT;
    depth_desc.BindFlags = D3D11_BIND_DEPTH_STENCIL;

    device->CreateTexture2D(&depth_desc, nullptr, depth_stencil_buffer.GetAddressOf());

    // 2️⃣ Create depth stencil view
    D3D11_DEPTH_STENCIL_VIEW_DESC dsv_desc = {};
    dsv_desc.Format = depth_desc.Format;
    dsv_desc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;

    device->CreateDepthStencilView(depth_stencil_buffer.Get(), &dsv_desc, depth_stencil_view.GetAddressOf());

    // 3️⃣ Bind depth buffer
    device_context->OMSetRenderTargets(1, render_target.GetAddressOf(), depth_stencil_view.Get());
}

bool renderer::init() {
    cleanup();

    if (!create_device_and_swap_chain()) return false;
    if (!create_render_target()) return false;
    create_depth_buffer(800, 600);
    create_viewport(800.0f, 600.0f);

    return true;
}

void renderer::render() {
    float clear_color[] = { 1.0f, 0.0f, 0.352941f, 1.0f };
    device_context->ClearRenderTargetView(render_target.Get(), clear_color);
    swap_chain->Present(1, 0);
}

void renderer::cleanup() {
    render_target.Reset();
    swap_chain.Reset();
    device_context.Reset();
    device.Reset();
}