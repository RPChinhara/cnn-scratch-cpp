#pragma once

#include <d3d11.h>
#include <wrl/client.h>  // For Microsoft::WRL::ComPtr

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")

class renderer {
public:
    renderer(HWND hwnd);
    ~renderer();

    bool init();
    void render();
    void cleanup();

private:
    HWND hwnd;

    Microsoft::WRL::ComPtr<ID3D11Device> device;
    Microsoft::WRL::ComPtr<ID3D11DeviceContext> device_context;
    Microsoft::WRL::ComPtr<IDXGISwapChain> swap_chain;
    Microsoft::WRL::ComPtr<ID3D11RenderTargetView> render_target;

    bool create_device_and_swap_chain();
    bool create_render_target();
};