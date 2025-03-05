#include "camera.h"
#include "input_handler.h"
#include "mesh.h"
#include "renderer.h"
#include "scene.h"
#include "tensor.h"
#include "window.h"

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    AllocConsole();
    freopen_s((FILE**)stdout, "CONOUT$", "w", stdout);

    window win(hInstance);
    renderer r(win.get_hwnd());
    camera cam;
    input_handler input;
    scene main_scene;

    if (!r.init())
        return -1;

    if (!main_scene.load(&r))
        return -1;

    while (win.process_messages()) {
        input.update(cam);
        main_scene.draw(r, cam);
    }

    return 0;
}