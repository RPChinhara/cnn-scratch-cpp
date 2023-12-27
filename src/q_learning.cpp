#include "q_learning.h"
#include "array.h"
#include "mathematics.h"

#include <iostream>
#include <random>

QLearning::QLearning(size_t n_states, size_t n_actions, float learning_rate, float discount_factor, float exploration_rate, float exploration_decay, float exploration_min)
{
    this->n_states = n_states;
    this->n_actions = n_actions;
    this->learning_rate = learning_rate;
    this->discount_factor = discount_factor;
    this->exploration_rate = exploration_rate;
    this->exploration_decay = exploration_decay;
    this->exploration_min = exploration_min;
    this->q_table = Zeros({ n_states, n_actions });
}

size_t QLearning::ChooseAction(size_t state)
{
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<> dis_1(0.0f, 1.0f);

    if (dis_1(rng) < exploration_rate) {
        std::uniform_int_distribution<> dis_2(0, n_actions - 1);
        return dis_2(rng);
    } else {
        Tensor sliced_q_table = Slice(q_table, state, 1);
        size_t max_idx = 0;
        float max = std::numeric_limits<float>::lowest();
        
        for (size_t i = 0; i < sliced_q_table.size; ++i) {
            if (sliced_q_table[i] > max) {
                max = sliced_q_table[i];
                max_idx = i;
            }
        }

        return max_idx;
    }
}

void QLearning::UpdateQtable(size_t state, size_t action, int reward, size_t next_state, bool done)
{
    Tensor sliced_q_table = Slice(q_table, next_state, 1);
    float next_max_q = std::numeric_limits<float>::lowest();

    for (size_t i = 0; i < sliced_q_table.size; ++i)
        if (sliced_q_table[i] > next_max_q)
            next_max_q = sliced_q_table[i];

    size_t idx = state == 0 ? action : (state * q_table.shape.back()) + action;
    q_table[idx] += learning_rate * (reward + discount_factor * next_max_q - q_table[idx]);

    std::cout << "q_table:               " << q_table << std::endl << std::endl;
    
    if (done)
        exploration_rate = 1.0f;

    if (exploration_rate > exploration_min)
        exploration_rate *= exploration_decay;
}