#include "mesh.h"

struct vertex {
    float x, y, z;     // position
    float r, g, b, a;  // color (optional)
};

bool mesh::initialize(renderer* r)
{
    vertex vertices[] = {
        {-0.5f,  0.5f, 0.0f},
        { 0.5f,  0.5f, 0.0f},
        {-0.5f, -0.5f, 0.0f},
        { 0.5f, -0.5f, 0.0f},
    };

    // Index Buffer for two triangles
    uint16_t indices[] = {0, 1, 2, 0, 2, 3};

    if (!r->create_vertex_buffer(&vertex_buffer, vertices, sizeof(vertex), 4))
        return false;

    // TODO: Optionally create an Index Buffer (if using indexed drawing).

    return true;
}

void mesh::render(renderer* r) {
    UINT stride = sizeof(vertex);
    UINT offset = 0;
    r->get_context()->IASetVertexBuffers(0, 1, &vertex_buffer, &stride, &offset);

    r->get_context()->Draw(4, 0);  // 4 vertices, starting at vertex 0
}