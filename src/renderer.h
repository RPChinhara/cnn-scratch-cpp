#pragma once

#include <d3d11.h>
#include <wrl/client.h>  // For Microsoft::WRL::ComPtr

class renderer {
public:
    renderer(HWND hwnd);
    ~renderer();

    bool init();
    void render();
    void cleanup();

private:
    HWND hwnd;

    Microsoft::WRL::ComPtr<ID3D11Device> device; // creates resources
    Microsoft::WRL::ComPtr<ID3D11DeviceContext> device_context; // tells the GPU what to do with the resources
    Microsoft::WRL::ComPtr<IDXGISwapChain> swap_chain; // handles the back buffer for double-buffered rendering
    Microsoft::WRL::ComPtr<ID3D11RenderTargetView> render_target; // A view that allows DirectX to draw to the back buffer

    bool create_device_and_swap_chain();
    bool create_render_target();
    void create_viewport(float window_width, float window_height);
};