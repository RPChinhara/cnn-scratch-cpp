#include "player.h"

bool mesh::initialize(renderer* r)
{
    struct Vertex {
        float x, y, z;  // Position
        float r, g, b;  // Color
    };

    Vertex vertices[] = {
        {-0.5f,  0.5f, 0.0f,  1, 0, 0},  // Top Left
        { 0.5f,  0.5f, 0.0f,  0, 1, 0},  // Top Right
        { 0.5f, -0.5f, 0.0f,  0, 0, 1},  // Bottom Right
        {-0.5f, -0.5f, 0.0f,  1, 1, 0},  // Bottom Left
    };

    // Index Buffer for two triangles
    uint16_t indices[] = {0, 1, 2, 0, 2, 3};

    if (!r->create_vertex_buffer(&vertex_buffer, vertices, sizeof(float) * 3, 3))
        return false;

    return true;
}