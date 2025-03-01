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

    mesh agent;
    if (!agent.init(&r))
        return -1;

    while (window.process_messages()) {
        r.begin_frame();
        agent.render(&r);
        r.end_frame();
    }

    return 0;
}