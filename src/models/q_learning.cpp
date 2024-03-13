#include "q_learning.h"
#include "arrays.h"
#include "tensor.h"

#include <random>

QLearning::QLearning(size_t n_states, size_t n_actions, float learning_rate, float discount_factor,
                     float exploration_rate, float exploration_decay, float exploration_min)
{
    this->n_states = n_states;
    this->n_actions = n_actions;
    this->learning_rate = learning_rate;
    this->discount_factor = discount_factor;
    this->exploration_rate = exploration_rate;
    this->exploration_decay = exploration_decay;
    this->exploration_min = exploration_min;
    q_table = Zeros({n_states, n_actions});
}

Action QLearning::ChooseAction(size_t state)
{
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<> dis1(0.0f, 1.0f);

    if (dis1(rng) < exploration_rate)
    {
        std::uniform_int_distribution<> dis2(0, n_actions - 1);
        return static_cast<Action>(dis2(rng));
    }
    else
    {
        Tensor sliced_q_table = Slice(q_table, state, 1);
        size_t max_idx = 0;
        float max = std::numeric_limits<float>::lowest();

        for (size_t i = 0; i < sliced_q_table.size; ++i)
        {
            if (sliced_q_table[i] > max)
            {
                max = sliced_q_table[i];
                max_idx = i;
            }
        }

        return static_cast<Action>(max_idx);
    }
}

void QLearning::UpdateQtable(size_t state, Action action, float reward, size_t next_state, bool done)
{
    std::cout << Slice(q_table, state, 1) << "\n\n";

    Tensor sliced_q_table = Slice(q_table, next_state, 1);
    float next_max_q = std::numeric_limits<float>::lowest();

    for (size_t i = 0; i < sliced_q_table.size; ++i)
        if (sliced_q_table[i] > next_max_q)
            next_max_q = sliced_q_table[i];

    size_t idx = state == 0 ? action : (state * q_table.shape.back()) + action;
    q_table[idx] += learning_rate * (reward + discount_factor * next_max_q - q_table[idx]);

    if (exploration_rate <= exploration_min || done)
        exploration_rate = 1.0f;

    if (exploration_rate > exploration_min)
        exploration_rate *= exploration_decay;
}