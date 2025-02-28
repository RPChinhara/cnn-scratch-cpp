#pragma once

#include <directxmath.h>
#include <d3d11.h>
#include <fstream>
#include <vector>
#include <wrl/client.h>

class renderer {
public:
    renderer(HWND hwnd);
    ~renderer();

    bool init();
    void cleanup();

    void begin_frame();
    void end_frame();

    bool create_vertex_buffer(ID3D11Buffer** buffer, const void* vertex_data, UINT vertex_size, UINT vertex_count);
    bool create_index_buffer(ID3D11Buffer** buffer, const uint32_t* index_data, UINT index_count);

    Microsoft::WRL::ComPtr<ID3D11DeviceContext> get_context();

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
    Microsoft::WRL::ComPtr<ID3D11InputLayout> input_layout;

    Microsoft::WRL::ComPtr<ID3D11Buffer> constant_buffer; // buffer that stores WVP matrix for the shader

    Microsoft::WRL::ComPtr<ID3D11RasterizerState> rasterizer_state;

    DirectX::XMMATRIX view_matrix;
    DirectX::XMMATRIX projection_matrix;

    bool create_device_and_swap_chain();
    bool create_render_target();
    bool create_depth_buffer(int width, int height);
    bool create_rasterizer_state();
    bool create_viewport(float window_width, float window_height);
    bool create_input_layout(const void* shader_bytecode, size_t bytecode_size);
    bool read_file(const std::string& filename, std::vector<char>& data);
    bool load_shaders();
};