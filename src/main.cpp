#include "arrs.h"
#include "logger.h"
#include "mesh.h"
#include "renderer.h"
#include "tensor.h"
#include "window.h"

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    logger::init();
    logger::log(fill({2, 3}, 1.0f));

    window window(hInstance);

    renderer r(window.get_hwnd());
    if (!r.init()) return -1;

    mesh floor = mesh(floor_vertices, std::size(floor_vertices), floor_indices, std::size(floor_indices));
    if (!floor.init(&r)) logger::log("Failed to init the floor");

    while (window.process_messages()) {
        r.begin_frame({floor});
        r.end_frame();
    }

    return 0;
}