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
    static int idx0 = 0; 
    static int idx1 = 0;
    static int idx2 = 0;
    static int idx3 = 0;

    auto b = dis_1(rng);
    std::cout << "dis_1: " << b << " exploration_rate: " << exploration_rate << std::endl;
    if (b < exploration_rate) {
        std::cout << "exploration" << std::endl;
        std::uniform_int_distribution<> dis_2(0, n_actions - 1);
        auto a = dis_2(rng);
        std::cout << "action: " << a << std::endl;

        if (a == 0) idx0 += 1;
        if (a == 1) idx1 += 1;
        if (a == 2) idx2 += 1;
        if (a == 3) idx3 += 1;

        std::cout << "idx 0: " << idx0 << " idx 1: " << idx1 << " idx 2: " << idx2 << " idx 3: " << idx3 << std::endl;
        return a;
    } else {
        Tensor sliced_q_table = Slice(q_table, state, 1);
        size_t max_idx = 0;
        float max = std::numeric_limits<float>::lowest();

        
        for (size_t i = 0; i < sliced_q_table.size; ++i) {
            std::cout << "sliced_q_table[i]: " << sliced_q_table[i] << " max: " << max << std::endl;
            if (sliced_q_table[i] > max) {
                std::cout << sliced_q_table[i] << " is bigger than " << max << std::endl;
                max = sliced_q_table[i];
                max_idx = i;
            }
        }
        
        std::cout << "max idx: " << max_idx << std::endl;
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

    std::cout << q_table << std::endl << std::endl;
    
    if (done)
        exploration_rate = 1.0f;

    if (exploration_rate > exploration_min)
        exploration_rate *= exploration_decay;
}