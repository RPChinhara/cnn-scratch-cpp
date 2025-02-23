#include "rand.h"
#include "renderer.h"
#include "tensor.h"
#include "window.h"

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    AllocConsole();
    freopen_s((FILE**)stdout, "CONOUT$", "w", stdout);
    freopen_s((FILE**)stderr, "CONOUT$", "w", stderr);

    window window(hInstance);

    renderer renderer;

    if (!renderer.init()) {
        return -1;
    }

    while (window.process_messages()) {
        renderer.render();
    }

    return 0;
}