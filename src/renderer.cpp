#include <iostream>

#include "renderer.h"

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

    HRESULT hr = D3D11CreateDeviceAndSwapChain(
        nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0,
        D3D11_SDK_VERSION, &sc_desc, swap_chain.GetAddressOf(),
        device.GetAddressOf(), nullptr, device_context.GetAddressOf()
    );

    if (FAILED(hr)) {
        std::cerr << "Failed to create Direct3D 11 device and swap.\n";
        return false;
    }

    return true;
}

bool renderer::create_render_target() {
    Microsoft::WRL::ComPtr<ID3D11Texture2D> back_buffer;
    HRESULT hr = swap_chain->GetBuffer(0, __uuidof(ID3D11Texture2D), (void**)back_buffer.GetAddressOf());
    if (FAILED(hr)) {
        std::cerr << "Failed to get buffer.\n";
        return false;
    }

    hr = device->CreateRenderTargetView(back_buffer.Get(), nullptr, render_target.GetAddressOf());
    if (FAILED(hr)) {
        std::cerr << "Failed to create render render target.\n";
        return false;
    }

    device_context->OMSetRenderTargets(1, render_target.GetAddressOf(), nullptr);
    return true;
}

bool renderer::init() {
    cleanup();

    if (!create_device_and_swap_chain()) return false;
    if (!create_render_target()) return false;
    
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