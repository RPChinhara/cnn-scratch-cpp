#include "q_learning.h"
#include "arrays.h"
#include "mathematics.h"

#include <random>

QLearning::QLearning(unsigned int n_states, unsigned int n_actions, float learning_rate, float discount_factor, float exploration_rate, float exploration_decay, float exploration_min) {
    this->n_states          = n_states;
    this->n_actions         = n_actions;
    this->learning_rate     = learning_rate;
    this->discount_factor   = discount_factor;
    this->exploration_rate  = exploration_rate;
    this->exploration_decay = exploration_decay;
    this->exploration_min   = exploration_min;
    this->q_table           = Zeros({ n_states, n_actions });
}

unsigned int QLearning::choose_action(unsigned int state) {
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<> dis_1(0.0f, 1.0f);

    if (dis_1(rng) < exploration_rate) {
        std::uniform_int_distribution<int> dis_2(0, n_actions - 1);
        return dis_2(rng);
    } else {
        Tensor sliced_q_table = Slice(q_table, state, 1);
        unsigned int max_idx = 0;
        unsigned int max = std::numeric_limits<unsigned int>::lowest();

        for (int i = 0; i < sliced_q_table._size; ++i) {
            if (sliced_q_table[i] > max) {
                max = sliced_q_table[i];
                max_idx = i;
            }
        }
            
        return max_idx;
    }
}

void QLearning::update_q_table(unsigned int state, unsigned int action, float reward, unsigned int next_state) {
    Tensor sliced_q_table = Slice(q_table, next_state, 1);
    float next_max_q = std::numeric_limits<float>::lowest();

    for (int i = 0; i < sliced_q_table._size; ++i)
        if (sliced_q_table[i] > next_max_q)
            next_max_q = sliced_q_table[i];

    unsigned int idx = state == 0 ? action : (state * q_table._shape.back()) + action;
    std::cout << "state: " << state << std::endl;
    std::cout << "idx: " << idx << std::endl;
    q_table[idx] += learning_rate * (reward + discount_factor * next_max_q - q_table[idx]);
        
    // if (exploration_rate > exploration_min)
    //     exploration_rate *= exploration_decay;
}