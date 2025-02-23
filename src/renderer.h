#pragma once

#include <d3d11.h>
#include <wrl/client.h>  // For Microsoft::WRL::ComPtr
#include <iostream>

class renderer {
public:
    renderer();
    ~renderer();

    bool init();
    void render();
    void shutdown();

private:
    Microsoft::WRL::ComPtr<ID3D11Device> device;
    Microsoft::WRL::ComPtr<ID3D11DeviceContext> device_context;
};