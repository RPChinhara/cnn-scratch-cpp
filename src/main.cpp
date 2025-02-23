#include "rand.h"
#include "renderer.h"
#include "tensor.h"
#include "window.h"

size_t synapse1 = 32;
size_t synapse2 = 32;
size_t synapse3 = 32;
size_t synapse4 = 32;
size_t synapse5 = 32;

tensor neuron1 = glorot_uniform({synapse1, synapse2});
tensor neuron2 = glorot_uniform({synapse2, synapse3});
tensor neuron3 = glorot_uniform({synapse3, synapse4});
tensor neuron4 = glorot_uniform({synapse4, synapse5});

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