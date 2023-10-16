#pragma once

#include "tensor.h"

class QLearning {
public:
    QLearning(unsigned int n_states, unsigned int n_actions, float learning_rate, float discount_factor, float exploration_rate, float exploration_decay, float exploration_min);
    unsigned int choose_action(unsigned int state);
    void update(unsigned int state, unsigned int action, float reward, float next_state);

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