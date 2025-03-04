#include "mesh.h"
#include "renderer.h"

mesh::mesh(const vertex* vertices, size_t vertex_count, const uint32_t* indices, size_t index_count) {
    this->vertices.assign(vertices, vertices + vertex_count);
    this->indices.assign(indices, indices + index_count);
}

bool mesh::init(renderer* r) {
    if (!r->create_vertex_buffer(vertex_buffer, vertices.data(), sizeof(vertex), vertices.size()))
        return false;

    if (!r->create_index_buffer(index_buffer, indices.data(), indices.size()))
        return false;

    return true;
}

void mesh::render(const Microsoft::WRL::ComPtr<ID3D11DeviceContext>& device_context) const {
    UINT stride = sizeof(vertex);
    UINT offset = 0;
    device_context->IASetVertexBuffers(0, 1, vertex_buffer.GetAddressOf(), &stride, &offset);
    device_context->IASetIndexBuffer(index_buffer.Get(), DXGI_FORMAT_R32_UINT, 0);

    device_context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);  // Use TriangleList for indexed geometry

    device_context->DrawIndexed(indices.size(), 0, 0);
}