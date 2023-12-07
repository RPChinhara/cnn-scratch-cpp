#pragma once

#include "tensor.h"

class QLearning
{
public:
    QLearning(unsigned int n_states, unsigned int n_actions, float learning_rate = 0.1f, float discount_factor = 0.9f, float exploration_rate = 0.1f, float exploration_decay = 0.995f, float exploration_min = 0.01f);
    unsigned int ChooseAction(unsigned int state);
    void UpdateQtable(unsigned int state, unsigned int action, float reward, unsigned int next_state);

    Tensor q_table;
private:
    unsigned int n_states;
    unsigned int n_actions;
    float learning_rate;
    float discount_factor;
    float exploration_rate;
    float exploration_decay;
    float exploration_min;
};