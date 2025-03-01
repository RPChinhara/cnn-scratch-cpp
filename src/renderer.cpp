#include <iostream>

#include "renderer.h"
#include "logger.h"

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

    return SUCCEEDED(hr);
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

bool renderer::create_depth_buffer(int width, int height) {
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

    HRESULT hr = device->CreateTexture2D(&depth_desc, nullptr, depth_stencil_buffer.GetAddressOf());
    if (FAILED(hr)) {
        std::cerr << "Failed to create texture2D.\n";
        return false;
    }

    // 2️⃣ Create depth stencil view - Converts the depth buffer texture into a depth-stencil view. This is necessary because DirectX doesn’t use raw textures directly—it needs views to read/write depth data.
    D3D11_DEPTH_STENCIL_VIEW_DESC dsv_desc = {};
    dsv_desc.Format = depth_desc.Format;
    dsv_desc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;

    hr = device->CreateDepthStencilView(depth_stencil_buffer.Get(), &dsv_desc, depth_stencil_view.GetAddressOf());
    if (FAILED(hr)) {
        std::cerr << "Failed to create depth stencil view.\n";
        return false;
    }

    // 3️⃣ Bind depth buffer to the pipeline
    device_context->OMSetRenderTargets(1, render_target.GetAddressOf(), depth_stencil_view.Get());
    return true;
}

bool renderer::create_depth_stencil_state() {
    D3D11_DEPTH_STENCIL_DESC dsDesc = {};
    dsDesc.DepthEnable = true;                          // 🔥 Enable depth testing
    dsDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL; // Write to depth buffer
    dsDesc.DepthFunc = D3D11_COMPARISON_LESS;           // Closer pixels pass
    dsDesc.StencilEnable = false;                       // Optional, if not using stencil

    HRESULT hr = device->CreateDepthStencilState(&dsDesc, depth_stencil_state.GetAddressOf());
    if (FAILED(hr)) {
        std::cerr << "Failed to create depth stencil state.\n";
        return false;
    }

    // Bind this state to the pipeline
    device_context->OMSetDepthStencilState(depth_stencil_state.Get(), 0);
    return true;
}

bool renderer::create_rasterizer_state() {
    D3D11_RASTERIZER_DESC rasterizer_desc = {};
    rasterizer_desc.FillMode = D3D11_FILL_SOLID;
    // rasterizer_desc.CullMode = D3D11_CULL_BACK;
    rasterizer_desc.CullMode = D3D11_CULL_NONE;  // Disable culling to see all faces
    rasterizer_desc.FrontCounterClockwise = FALSE;  // For clockwise winding

    HRESULT hr = device->CreateRasterizerState(&rasterizer_desc, rasterizer_state.GetAddressOf());
    if (FAILED(hr)) {
        return false;
    }

    return true;
}

bool renderer::create_viewport(float window_width, float window_height) {
    // NOTE:  Defines the area where DirectX will draw graphics inside the window because by default, DirectX does not know where to draw. We need to tell it how large the rendering area is.
    D3D11_VIEWPORT viewport = {};
    viewport.Width = window_width;
    viewport.Height = window_height;
    viewport.MinDepth = 0.0f;  // Closest depth (near plane)
    viewport.MaxDepth = 1.0f;  // Farthest depth (far plane)
    viewport.TopLeftX = 0;
    viewport.TopLeftY = 0;

    device_context->RSSetViewports(1, &viewport);
    return true;
}

bool renderer::create_input_layout(const void* shader_bytecode, size_t bytecode_size) {
    // NOTE: DirectX needs to know how to interpret vertex data (like position, color, texture coordinates). This is called an Input Layout.

    D3D11_INPUT_ELEMENT_DESC layout_desc[] = {
        {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0}
    };

    if (FAILED(device->CreateInputLayout(
        layout_desc,
        ARRAYSIZE(layout_desc),
        shader_bytecode,
        bytecode_size,
        input_layout.ReleaseAndGetAddressOf()
    ))) {
        return false;
    }

    return true;
}

bool renderer::read_file(const std::string& filename, std::vector<char>& data) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        return false;
    }
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    data.resize(file_size);
    file.read(data.data(), file_size);
    return true;
}

bool renderer::load_shaders() {
    // Load compiled vertex shader
    std::vector<char> vs_data;
    if (!read_file("shaders/compiled/basic_vs.cso", vs_data)) {
        return false;
    }

    HRESULT hr = device->CreateVertexShader(vs_data.data(), vs_data.size(), nullptr, vertex_shader.GetAddressOf());
    if (FAILED(hr)) {
        return false;
    }

    // Create input layout using the vertex shader blob data
    if (!create_input_layout(vs_data.data(), vs_data.size()))
    return false;

    // Load compiled pixel shader
    std::vector<char> ps_data;
    if (!read_file("shaders/compiled/basic_ps.cso", ps_data)) {
        return false;
    }

    hr = device->CreatePixelShader(ps_data.data(), ps_data.size(), nullptr, pixel_shader.GetAddressOf());
    if (FAILED(hr)) {
        return false;
    }

    // Bind shaders to pipeline (you usually do this before drawing, but for now just do it once)
    device_context->VSSetShader(vertex_shader.Get(), nullptr, 0);
    device_context->PSSetShader(pixel_shader.Get(), nullptr, 0);

    return true;
}

bool renderer::init() {
    cleanup();

    if (!create_device_and_swap_chain())
        return false;
    if (!create_render_target())
        return false;
    if (!create_depth_buffer(800, 600))
        return false;
    if (!create_depth_stencil_state())
        return false;
    if (!create_rasterizer_state())
        return false;
    if (!create_viewport(800.0f, 600.0f))
        return false;
    if (!load_shaders())
        return false;
    if (!create_constant_buffer(&constant_buffer))
        return false;

    // Set up the camera (view matrix)
    view_matrix = DirectX::XMMatrixLookAtLH(
        {0.0f, 0.0f, -3.0f},  // Camera position (behind the object)
        {0.0f, 0.0f, 0.0f},   // Looking at origin
        {0.0f, 1.0f, 0.0f}    // Up vector
    );

    // Set projection matrix
    projection_matrix = DirectX::XMMatrixPerspectiveFovLH(
        DirectX::XMConvertToRadians(60.0f), // FOV
        800.0f / 600.0f,                    // Aspect ratio
        0.1f, 100.0f                        // Near & far planes
    );

    return true;
}

bool renderer::create_vertex_buffer(ID3D11Buffer** buffer, const void* vertex_data, UINT vertex_size, UINT vertex_count) {
    D3D11_BUFFER_DESC buffer_desc = {};
    buffer_desc.Usage = D3D11_USAGE_DEFAULT;
    buffer_desc.ByteWidth = vertex_size * vertex_count;
    buffer_desc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    buffer_desc.CPUAccessFlags = 0;

    D3D11_SUBRESOURCE_DATA init_data = {};
    init_data.pSysMem = vertex_data;

    HRESULT hr = device->CreateBuffer(&buffer_desc, &init_data, buffer);
    if (FAILED(hr)) {
        logger::log("Failed to create buffer for vertex");
        return false;
    }

    return true;
}

bool renderer::create_index_buffer(ID3D11Buffer** buffer, const uint32_t* index_data, UINT index_count) {
    D3D11_BUFFER_DESC buffer_desc = {};
    buffer_desc.Usage = D3D11_USAGE_DEFAULT;
    buffer_desc.ByteWidth = sizeof(uint32_t) * index_count;
    buffer_desc.BindFlags = D3D11_BIND_INDEX_BUFFER;
    buffer_desc.CPUAccessFlags = 0;

    D3D11_SUBRESOURCE_DATA init_data = {};
    init_data.pSysMem = index_data;

    HRESULT hr = device->CreateBuffer(&buffer_desc, &init_data, buffer);
    if (FAILED(hr)) {
        logger::log("Failed to create buffer for index");
        return false;
    }

    return true;
}

bool renderer::create_constant_buffer(ID3D11Buffer** buffer) {
    D3D11_BUFFER_DESC cbd = {};
    cbd.Usage = D3D11_USAGE_DEFAULT;
    cbd.ByteWidth = sizeof(DirectX::XMMATRIX);
    cbd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    cbd.CPUAccessFlags = 0;

    HRESULT hr = device->CreateBuffer(&cbd, nullptr, constant_buffer.GetAddressOf());
    if (FAILED(hr)) {
        // Handle error (log, assert, etc.)
    }
    return true;
}

Microsoft::WRL::ComPtr<ID3D11DeviceContext> renderer::get_context() {
    return device_context;
}

void renderer::begin_frame() {
    // 1️⃣ Clear the render target (color buffer)
    float clear_color[4] = { 1.0f, 1.0f, 1.0f, 1.0f };  // white background
    device_context->ClearRenderTargetView(render_target.Get(), clear_color);

    // 2️⃣ Clear the depth/stencil buffer
    device_context->ClearDepthStencilView(depth_stencil_view.Get(), D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 1.0f, 0);  // 1.0 = farthest depth (default clear)

    // 3️⃣ Set the input layout (before drawing anything, ensures vertex format matches shader expectations)
    device_context->IASetInputLayout(input_layout.Get());

    // 4️⃣ Set the render target and depth buffer again (this is optional if you are 100% sure they didn’t change between frames)
    device_context->OMSetRenderTargets(1, render_target.GetAddressOf(), depth_stencil_view.Get());

    device_context->RSSetState(rasterizer_state.Get());

    // World Matrix
    DirectX::XMMATRIX world_matrix = DirectX::XMMatrixIdentity();  // Start with identity (no transform)

    // Example: rotate it slowly
    static float angle = 0.0f;
    angle += 0.01f;
    world_matrix = DirectX::XMMatrixRotationY(angle);

    // Combine World * View * Projection into final matrix
    DirectX::XMMATRIX wvp = world_matrix * view_matrix * projection_matrix;

    // Upload this WVP matrix to the vertex shader constant buffer
    device_context->UpdateSubresource(constant_buffer.Get(), 0, nullptr, &wvp, 0, 0);

    // Set the constant buffer to the vertex shader
    device_context->VSSetConstantBuffers(0, 1, constant_buffer.GetAddressOf());

    // 5️⃣ Now ready to issue draw calls (Draw, DrawIndexed, etc.)
}

void renderer::end_frame() {
    swap_chain->Present(1, 0);
}

void renderer::cleanup() {
    render_target.Reset();
    swap_chain.Reset();
    device_context.Reset();
    device.Reset();
}