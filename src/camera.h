#pragma once

#include <directxmath.h>

struct camera {
    DirectX::XMVECTOR position = DirectX::XMVectorSet(0.0f, 40.0f, -40.0f, 1.0f); // Start further back
    DirectX::XMVECTOR target = DirectX::XMVectorSet(0.0f, 0.0f, 0.0f, 1.0f);      // Looking at origin
    DirectX::XMVECTOR up = DirectX::XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f);

    void move(float dx, float dz) {
        position = DirectX::XMVectorAdd(position, DirectX::XMVectorSet(dx, 0.0f, dz, 0.0f));
        target = DirectX::XMVectorAdd(target, DirectX::XMVectorSet(dx, 0.0f, dz, 0.0f));
    }

    DirectX::XMMATRIX get_view_matrix() const {
        return DirectX::XMMatrixLookAtLH(position, target, up);
    }
};