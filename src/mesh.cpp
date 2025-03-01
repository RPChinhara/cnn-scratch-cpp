#include "mesh.h"
#include "logger.h"

struct vertex {
    float x, y, z;
};

bool mesh::init(renderer* r) {
    vertex floor_vertices[] = {
        { -0.5f, 0.0f, -0.5f }, // Bottom left
        { -0.5f, 0.0f,  0.5f }, // Top left
        {  0.5f, 0.0f, -0.5f }, // Bottom right
        {  0.5f, 0.0f,  0.5f }  // Top right
    };

    vertex cube_vertices[] = {
        // Front face
        { -0.5f, -0.5f, -0.5f }, // bottom-left-front
        { -0.5f,  0.5f, -0.5f }, // top-left-front
        {  0.5f, -0.5f, -0.5f }, // bottom-right-front
        {  0.5f,  0.5f, -0.5f }, // top-right-front

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

    uint32_t floor_indices[] = {
        0, 1, 2, 2, 1, 3  // Two triangles forming a quad
    };

    vertex_count = ARRAYSIZE(cube_vertices);
    index_count = ARRAYSIZE(indices);

    if (!r->create_vertex_buffer(&vertex_buffer, cube_vertices, sizeof(vertex), vertex_count)) {
        logger::log("Failed to create vertex buffer");
        return false;
    }

    if (!r->create_index_buffer(&index_buffer, indices, ARRAYSIZE(indices))) {
        logger::log("Failed to create index buffer");
        return false;
    }

    vertex_count2 = ARRAYSIZE(floor_vertices);
    index_count2 = ARRAYSIZE(floor_indices);

    if (!r->create_vertex_buffer(&vertex_buffer2, floor_vertices, sizeof(vertex), vertex_count2)) {
        logger::log("Failed to create vertex buffer");
        return false;
    }

    if (!r->create_index_buffer(&index_buffer2, floor_indices, ARRAYSIZE(floor_indices))) {
        logger::log("Failed to create index buffer");
        return false;
    }

    return true;
}

void mesh::render(renderer* r) {
    auto context = r->get_context();

    UINT stride = sizeof(vertex);
    UINT offset = 0;
    context->IASetVertexBuffers(0, 1, vertex_buffer.GetAddressOf(), &stride, &offset);
    context->IASetIndexBuffer(index_buffer.Get(), DXGI_FORMAT_R32_UINT, 0);

    // context->IASetVertexBuffers(0, 1, vertex_buffer2.GetAddressOf(), &stride, &offset);
    // context->IASetIndexBuffer(index_buffer2.Get(), DXGI_FORMAT_R32_UINT, 0);

    // context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
    context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);  // Use TriangleList for indexed geometry

    context->DrawIndexed(index_count, 0, 0);  // 36 for the cube
    // context->DrawIndexed(index_count2, 0, 0);
    // context->Draw(vertex_count, 0);
}