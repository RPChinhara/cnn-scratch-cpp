#include "q_learning.h"
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
    // Explore
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis_1(0.0f, 1.0f);

    Tensor xx = uniform_distribution({ 2, 3 }, 0.0f, 1.0f);
    std::cout << xx << std::endl;
    auto df = argmax(xx);
    std::cout << df << std::endl;
    if (dis_1(gen) > exploration_rate) {
        auto df = argmax(xx);
        std::cout << df << std::endl;
        return 3;
    }

    // Exploit
    std::uniform_int_distribution<int> dis_2(0, n_actions - 1);
    return dis_2(gen);
}

void QLearning::update(unsigned int state, unsigned int action, float reward, float next_state) {
    // # Q-learning update rule
    //     best_next_action = np.argmax(self.q_table[next_state, :])
    //     q_target = reward + self.discount_factor * self.q_table[next_state, best_next_action]
    //     q_delta = q_target - self.q_table[state, action]
    //     self.q_table[state, action] += self.learning_rate * q_delta
        
    //     # Exploration rate decay
    //     if self.exploration_rate > self.exploration_min:
    //         self.exploration_rate *= self.exploration_decay
}