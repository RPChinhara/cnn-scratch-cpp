#include "camera.h"
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
    scene main_scene;

    if (!r.init())
        return -1;

    if (!main_scene.load(&r))
        return -1;

    while (win.process_messages()) {
        win.update_camera(cam);
        win.handle_mouse(cam);
        main_scene.draw(r, cam);
    }

    return 0;
}