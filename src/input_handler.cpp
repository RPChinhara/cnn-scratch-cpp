#include "input_handler.h"
#include "camera.h"

input_handler::input_handler() {
    GetCursorPos(&last_mouse_pos);  // Initialize mouse position at startup
}

void input_handler::update(camera& cam) {
    handle_keyboard(cam);
    handle_mouse(cam);
}

void input_handler::handle_keyboard(camera& cam) {
    const float speed = 0.1f;

    if (GetAsyncKeyState(0x41)) cam.move( speed, 0.0f);  // A - left
    if (GetAsyncKeyState(0x44)) cam.move(-speed, 0.0f);  // D - right
    if (GetAsyncKeyState(0x57)) cam.move(0.0f,  speed);  // W - forward
    if (GetAsyncKeyState(0x53)) cam.move(0.0f, -speed);  // S - backward
}

void input_handler::handle_mouse(camera& cam) {
    POINT current_mouse_pos;
    GetCursorPos(&current_mouse_pos);

    float delta_x = (current_mouse_pos.x - last_mouse_pos.x) * 0.002f;
    float delta_y = (current_mouse_pos.y - last_mouse_pos.y) * 0.002f;

    cam.rotate(delta_x, -delta_y);  // Invert Y for natural look controls

    last_mouse_pos = current_mouse_pos;
}