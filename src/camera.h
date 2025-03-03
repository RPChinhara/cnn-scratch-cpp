#pragma once

#include <directxmath.h>

struct camera {
    DirectX::XMVECTOR position = DirectX::XMVectorSet(0.0f, 5.0f, -15.0f, 1.0f);
    float yaw = 0.0f;   // Left/Right
    float pitch = 0.0f; // Up/Down

    DirectX::XMVECTOR get_forward() const {
        float y = sinf(pitch);
        float r = cosf(pitch);
        float x = r * sinf(yaw);
        float z = r * cosf(yaw);
        return DirectX::XMVectorSet(x, y, z, 0.0f);
    }

    DirectX::XMMATRIX get_view_matrix() const {
        auto forward = get_forward();
        auto target = DirectX::XMVectorAdd(position, forward);
        auto up = DirectX::XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f);
        return DirectX::XMMatrixLookAtLH(position, target, up);
    }

    void move(float dx, float dz) {
        auto forward = get_forward();
        auto right = DirectX::XMVector3Cross(forward, DirectX::XMVectorSet(0, 1, 0, 0));

        position = DirectX::XMVectorAdd(position, DirectX::XMVectorScale(right, dx));
        position = DirectX::XMVectorAdd(position, DirectX::XMVectorScale(forward, dz));
    }

    void rotate(float delta_yaw, float delta_pitch) {
        yaw += delta_yaw;
        pitch += delta_pitch;

        // Clamp pitch to avoid looking straight up/down (Gimbal lock prevention)
        const float limit = DirectX::XM_PIDIV2 - 0.01f;
        if (pitch > limit) pitch = limit;
        if (pitch < -limit) pitch = -limit;
    }
};