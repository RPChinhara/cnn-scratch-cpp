#pragma once

#include "tensor.h"

class QLearning
{
public:
    QLearning(size_t n_states, size_t n_actions, float learning_rate = 0.1f, float discount_factor = 0.9f, float exploration_rate = 0.1f, float exploration_decay = 0.995f, float exploration_min = 0.01f);
    size_t ChooseAction(size_t state);
    void UpdateQtable(size_t state, size_t action, int reward, size_t next_state);
private:
    Tensor q_table;
    size_t n_states;
    size_t n_actions;
    float learning_rate;
    float discount_factor;
    float exploration_rate;
    float exploration_decay;
    float exploration_min;
};