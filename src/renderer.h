#pragma once

#include <directxmath.h>
#include <d3d11.h>
#include <string>
#include <vector>
#include <wrl/client.h>

#include "camera.h"

class mesh;

struct constant_buffer_data {
    DirectX::XMMATRIX wvp;
    DirectX::XMFLOAT4 objectColor;
};

class renderer {
public:
    renderer(HWND hwnd);
    ~renderer();

    bool init();
    void cleanup();

    void begin_frame(const std::vector<mesh>& meshes, const camera& cam);
    void end_frame();

    bool create_vertex_buffer(Microsoft::WRL::ComPtr<ID3D11Buffer>& buffer, const void* vertex_data, UINT vertex_size, UINT vertex_count);
    bool create_index_buffer(Microsoft::WRL::ComPtr<ID3D11Buffer>& buffer, const uint32_t* index_data, UINT index_count);
    bool create_constant_buffer(Microsoft::WRL::ComPtr<ID3D11Buffer>& buffer);

private:
    HWND hwnd;

    Microsoft::WRL::ComPtr<ID3D11Device> device;
    Microsoft::WRL::ComPtr<ID3D11DeviceContext> device_context;

    Microsoft::WRL::ComPtr<IDXGISwapChain> swap_chain;

    Microsoft::WRL::ComPtr<ID3D11RenderTargetView> render_target;

    Microsoft::WRL::ComPtr<ID3D11DepthStencilView> depth_stencil_view;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> depth_stencil_buffer;
    Microsoft::WRL::ComPtr<ID3D11DepthStencilState> depth_stencil_state;

    Microsoft::WRL::ComPtr<ID3D11VertexShader> vertex_shader;
    Microsoft::WRL::ComPtr<ID3D11PixelShader> pixel_shader;
    Microsoft::WRL::ComPtr<ID3D11InputLayout> input_layout;

    Microsoft::WRL::ComPtr<ID3D11Buffer> constant_buffer;

    Microsoft::WRL::ComPtr<ID3D11RasterizerState> rasterizer_state;

    DirectX::XMMATRIX view_matrix;
    DirectX::XMMATRIX projection_matrix;

    bool create_device_and_swap_chain();
    bool create_render_target();
    bool create_depth_buffer(int width, int height);
    bool create_depth_stencil_state();
    bool create_rasterizer_state();
    bool create_viewport(float window_width, float window_height);
    bool create_input_layout(const void* shader_bytecode, size_t bytecode_size);
    bool load_shaders();
};