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

    renderer renderer(window.get_hwnd());
    if (!renderer.init()) return -1;

    mesh agent;
    if (!agent.initialize(&renderer))
        return -1;

    while (window.process_messages()) {
        renderer.render();

        agent.render(&renderer);
    }

    return 0;
}