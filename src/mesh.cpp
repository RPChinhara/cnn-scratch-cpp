#include "mesh.h"

struct vertex {
    float x, y, z;     // position
    float r, g, b, a;  // color (optional)
};

bool mesh::initialize(renderer* r)
{
    vertex rect_vertices[] = {
        {-0.5f, -0.5f, 0.0f}, // Bottom left
        {-0.5f, 0.5f, 0.0f},  // Bottom right
        {0.5f,  -0.5f, 0.0f},  // Top right
        {0.5f,  0.5f, 0.0f},  // Top right
    };

    vertex cube_vertices[] = {
        // Front face
        { -0.5f, -0.5f, -0.5f },  // bottom-left-front
        { -0.5f,  0.5f, -0.5f },  // top-left-front
        {  0.5f, -0.5f, -0.5f },  // bottom-right-front
        {  0.5f,  0.5f, -0.5f },  // top-right-front

        // Back face
        { -0.5f, -0.5f,  0.5f },
        { -0.5f,  0.5f,  0.5f },
        {  0.5f, -0.5f,  0.5f },
        {  0.5f,  0.5f,  0.5f }
    };

    uint32_t indices[] = {
        0, 1, 2,  1, 3, 2, // Front
        4, 6, 5,  5, 6, 7, // Back
        0, 2, 4,  4, 2, 6, // Bottom
        1, 5, 3,  3, 5, 7, // Top
        0, 4, 1,  1, 4, 5, // Left
        2, 3, 6,  6, 3, 7  // Right
    };

    vertex_count = ARRAYSIZE(rect_vertices);

    if (!r->create_vertex_buffer(&vertex_buffer, rect_vertices, sizeof(vertex), vertex_count))
        return false;

    // TODO: Optionally create an Index Buffer (if using indexed drawing).

    return true;
}

void mesh::render(renderer* r) {
    auto context = r->get_context();

    UINT stride = sizeof(vertex);
    UINT offset = 0;
    context->IASetVertexBuffers(0, 1, vertex_buffer.GetAddressOf(), &stride, &offset);
    context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);

    context->Draw(vertex_count, 0);
}