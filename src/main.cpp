#include "arrs.h"
#include "camera.h"
#include "input_handler.h"
#include "logger.h"
#include "mesh.h"
#include "renderer.h"
#include "scene.h"
#include "tensor.h"
#include "window.h"

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    logger::init();
    logger::log(fill({2, 3}, 1.0f));

    window window(hInstance);

    renderer r(window.get_hwnd());
    if (!r.init()) return -1;

    input_handler input;
    camera cam;

    scene main_scene;
    if (!main_scene.load(&r)) return -1;

    while (window.process_messages()) {
        input.update(cam);
        main_scene.draw(r, cam);
    }

    return 0;
}