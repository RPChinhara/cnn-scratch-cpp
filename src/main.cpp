#include "arrs.h"
#include "camera.h"
#include "logger.h"
#include "mesh.h"
#include "renderer.h"
#include "tensor.h"
#include "window.h"

static POINT last_mouse_pos = {};

void handle_mouse(camera& cam) {
    POINT current_mouse_pos;
    GetCursorPos(&current_mouse_pos);

    float delta_x = (current_mouse_pos.x - last_mouse_pos.x) * 0.002f;
    float delta_y = (current_mouse_pos.y - last_mouse_pos.y) * 0.002f;

    cam.rotate(delta_x, -delta_y);  // Invert Y for natural look controls

    last_mouse_pos = current_mouse_pos;  // Update for next frame
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    logger::init();
    logger::log(fill({2, 3}, 1.0f));

    window window(hInstance);

    renderer r(window.get_hwnd());
    if (!r.init()) return -1;

    camera cam;

    mesh floor = mesh(floor_vertices, std::size(floor_vertices), floor_indices, std::size(floor_indices));
    if (!floor.init(&r)) logger::log("Failed to init the floor");

    mesh agent(cube_vertices, std::size(cube_vertices), cube_indices, std::size(cube_indices));
    if (!agent.init(&r)) logger::log("Failed to init the agent");

    while (window.process_messages()) {
        if (GetAsyncKeyState(0x41))
            cam.move(0.1f, 0.0f);
        if (GetAsyncKeyState(0x44))
            cam.move(-0.1f, 0.0f);
        if (GetAsyncKeyState(0x57))
            cam.move(0.0f, 0.1f);
        if (GetAsyncKeyState(0x53))
            cam.move(0.0f, -0.1f);
        handle_mouse(cam);

        r.begin_frame({floor, agent}, cam);
        r.end_frame();
    }

    return 0;
}