#include "physics.h"

void CheckBoundaryCollision(RECT& rect, int window_width, int window_height) {
    if (rect.left < 0) {
        rect.left = 0;
        // rect.right = rect.left + (rect.right - rect.left);
        rect.right = 50;
    }
    if (rect.top < 0) {
        rect.top = 0;
        // rect.bottom = rect.top + (rect.bottom - rect.top);
        rect.bottom = 50;
    }
    if (rect.right > 1904) {
        rect.right = 1904;
        // rect.left = rect.right - (rect.right - rect.left);
        rect.left = 1854;
    }
    if (rect.bottom > 1041) {
        rect.bottom = 1041;
        // rect.top = rect.bottom - (rect.bottom - rect.top);
        rect.top = 991;
    }
}