#include "tensor.h"

// NOTE: Use this style instead of weights or w this is more intuitive?
// TODO: Consider graph-based AI models instead of just dense layers as it more similar to how neurons in brain are connected?
size_t synapse1 = 32;
size_t synapse2 = 32;
size_t synapse3 = 32;
size_t synapse4 = 32;
size_t synapse5 = 32;

tensor neuron1 = glorot_uniform({synapse1, synapse2});
tensor neuron2 = glorot_uniform({synapse2, synapse3});
tensor neuron3 = glorot_uniform({synapse3, synapse4});
tensor neuron4 = glorot_uniform({synapse4, synapse5});

int main() {

    return 0;
}