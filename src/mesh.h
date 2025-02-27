
#pragma once

#include <d3d11.h>
#include <wrl/client.h>

#include "renderer.h"

class mesh
{
public:
    bool initialize(renderer* r);
    void render(renderer* r);

private:
    Microsoft::WRL::ComPtr<ID3D11Buffer> vertex_buffer;
    UINT vertex_count = 0;
};