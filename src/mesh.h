
#pragma once

#include <d3d11.h>
#include <wrl/client.h>

#include "renderer.h"

class mesh {
public:
    bool init(renderer* r);
    void render(renderer* r);

private:
    Microsoft::WRL::ComPtr<ID3D11Buffer> vertex_buffer;
    Microsoft::WRL::ComPtr<ID3D11Buffer> index_buffer;

    Microsoft::WRL::ComPtr<ID3D11Buffer> vertex_buffer2;
    Microsoft::WRL::ComPtr<ID3D11Buffer> index_buffer2;

    UINT vertex_count = 0;
    uint32_t index_count = 0;

    UINT vertex_count2 = 0;
    uint32_t index_count2 = 0;
};