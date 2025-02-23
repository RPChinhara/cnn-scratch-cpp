#include "arrs.h"
#include "logger.h"
#include "renderer.h"
#include "window.h"
#include "tensor.h"

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    logger::init();
    logger::log(fill({2, 3}, 1.0f));

    window window(hInstance);

    renderer renderer;

    if (!renderer.init())
        return -1;

    while (window.process_messages()) {
        renderer.render();
    }

    return 0;
}