#pragma once

#include <d3d11.h>
#include <wrl/client.h>  // For Microsoft::WRL::ComPtr
#include <iostream>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")

class renderer {
public:
    renderer() = default;
    ~renderer();

    bool init();
    void render();
    void shutdown();

private:
    Microsoft::WRL::ComPtr<ID3D11Device> device;
    Microsoft::WRL::ComPtr<ID3D11DeviceContext> device_context;
};