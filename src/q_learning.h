#pragma once

#include "tensor.h"

class QLearning {
public:
    QLearning(unsigned int n_states, unsigned int n_actions, float learning_rate = 0.01f, float discount_factor = 0.95, float exploration_rate = 1.0f, float exploration_decay = 0.995f, float exploration_min = 0.01f);
    unsigned int choose_action(unsigned int state);
    void update(unsigned int state, unsigned int action, float reward, unsigned int next_state);

private:
    unsigned int n_states;
    unsigned int n_actions;
    float learning_rate;
    float discount_factor;
    float exploration_rate;
    float exploration_decay;
    float exploration_min;
    Tensor q_table;
};