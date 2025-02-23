#include "logger.h"
#include "renderer.h"
#include "tensor.h"
#include "window.h"

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    logger::initialize();

    window window(hInstance);

    renderer renderer;

    if (!renderer.init())
        return -1;

    while (window.process_messages()) {
        renderer.render();
    }

    return 0;
}