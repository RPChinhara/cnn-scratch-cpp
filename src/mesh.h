
#pragma once

#include <d3d11.h>
#include <wrl/client.h>

#include "renderer.h"

struct vertex {
    float x, y, z;
};

class mesh {
public:
    mesh() = default;
    mesh(const vertex* vertices, size_t vertex_count, const uint32_t* indices, size_t index_count);
    bool init(renderer* r);
    void render(const Microsoft::WRL::ComPtr<ID3D11DeviceContext>& device_context) const;

private:
    Microsoft::WRL::ComPtr<ID3D11Buffer> vertex_buffer;
    Microsoft::WRL::ComPtr<ID3D11Buffer> index_buffer;

    std::vector<vertex> vertices;
    std::vector<uint32_t> indices;
};