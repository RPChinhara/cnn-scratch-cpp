#pragma once

#include <windows.h>

class camera;

class input_handler {
public:
    input_handler();

    void update(camera& cam);

private:
    POINT last_mouse_pos;

    void handle_keyboard(camera& cam);
    void handle_mouse(camera& cam);
};