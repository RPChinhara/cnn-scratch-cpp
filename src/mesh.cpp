#include "mesh.h"

struct vertex {
    float x, y, z;     // position
    float r, g, b, a;  // color (optional)
};

bool mesh::initialize(renderer* r)
{
    vertex vertices[] = {
        {-0.5f, -0.5f, 0.0f}, // Bottom left
        {0.5f, -0.5f, 0.0f},  // Bottom right
        {0.5f,  0.5f, 0.0f},  // Top right

        {-0.5f, -0.5f, 0.0f}, // Bottom left
        {0.5f,  0.5f, 0.0f},  // Top right
        {-0.5f,  0.5f, 0.0f}  // Top left
    };

    vertex_count = ARRAYSIZE(vertices);

    if (!r->create_vertex_buffer(&vertex_buffer, vertices, sizeof(vertex), vertex_count))
        return false;

    // TODO: Optionally create an Index Buffer (if using indexed drawing).

    return true;
}

void mesh::render(renderer* r) {
    auto context = r->get_context();

    UINT stride = sizeof(vertex);
    UINT offset = 0;
    context->IASetVertexBuffers(0, 1, vertex_buffer.GetAddressOf(), &stride, &offset);
    context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

    context->Draw(vertex_count, 0);
}