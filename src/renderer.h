#pragma once

#include <d3d11.h>
#include <fstream>
#include <vector>
#include <wrl/client.h>

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

    Microsoft::WRL::ComPtr<ID3D11DepthStencilView> depth_stencil_view;  // Depth buffer
    Microsoft::WRL::ComPtr<ID3D11Texture2D> depth_stencil_buffer;
    Microsoft::WRL::ComPtr<ID3D11DepthStencilState> depth_stencil_state;

    Microsoft::WRL::ComPtr<ID3D11VertexShader> vertex_shader; // processes each vertex (position, color, etc.)
    Microsoft::WRL::ComPtr<ID3D11PixelShader> pixel_shader; // decides what color each pixel should be

    bool create_device_and_swap_chain();
    bool create_render_target();
    bool create_depth_buffer(int width, int height);
    void create_viewport(float window_width, float window_height);
    bool read_file(const std::string& filename, std::vector<char>& data);
    bool load_shaders();
};