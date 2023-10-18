#include "q_learning.h"
#include "arrays.h"
#include "initializers.h"
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
    this->q_table           = zeros({ n_states, n_actions });
}

unsigned int QLearning::choose_action(unsigned int state) {
    // TODO: Use epsilon-greedy policy or other policies?
    // Exploit
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis_1(0.0f, 1.0f);

    if (dis_1(gen) > exploration_rate) {
        Tensor sliced_q_table = slice(q_table, state, 1);
    	unsigned int max = std::numeric_limits<unsigned int>::lowest();
        for(int i = 0; i < sliced_q_table._size; ++i)
            if (sliced_q_table[i] > max)
                max = sliced_q_table[i];
            
        return max;
    }

    // Explore
    std::uniform_int_distribution<int> dis_2(0, n_actions - 1);
    return dis_2(gen);
}

void QLearning::update(unsigned int state, unsigned int action, float reward, unsigned int next_state) {
    // Q-learning update rule -> Q(s, a) = Q(s, a) + α * [R + γ * max(Q(s', a')) - Q(s, a)]
    // Q(s, a) is the Q-value of state-action pair (s, a).
    // α is the learning rate, which controls how much the Q-values are updated based on new information.
    // R is the immediate reward received after taking action a in state s.
    // γ is the discount factor, which determines the importance of future rewards.
    // s' is the next state after taking action a.
    // a' is the action that maximizes the Q-value in the next state s'.

    Tensor sliced_q_table = slice(q_table, next_state, 1);
    unsigned int max = std::numeric_limits<unsigned int>::lowest();
    for(int i = 0; i < sliced_q_table._size; ++i)
        if (sliced_q_table[i] > max)
            max = sliced_q_table[i];

    unsigned int best_next_action = max;
    unsigned int idx = best_next_action ? next_state == 0 : (next_state * q_table._shape.back()) + best_next_action;
    float q_target = reward + discount_factor * q_table[idx];
    idx = action ? state == 0 : (state * q_table._shape.back()) + action;
    float q_delta = q_target - q_table[idx];
    q_table[idx] += learning_rate * q_delta;
        
    // Exploration rate decay
    if (exploration_rate > exploration_min)
        exploration_rate *= exploration_decay;
}