#pragma once

#include "action.h"
#include "tensor.h"

class QLearning
{
  public:
    QLearning(size_t n_states, size_t n_actions, float learning_rate = 0.01f, float discount_factor = 0.5f,
              float exploration_rate = 1.0f, float exploration_decay = 0.995f, float exploration_min = 0.01f);
    Action ChooseAction(size_t state);
    void UpdateQtable(size_t state, Action action, float reward, size_t next_state, bool done);

    float exploration_rate;

  private:
    Tensor q_table;
    size_t n_states;
    size_t n_actions;
    float learning_rate;
    float discount_factor;
    float exploration_decay;
    float exploration_min;
};