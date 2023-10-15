#include "q_learning.h"
#include "initializers.h"

QLearning::QLearning(unsigned int n_states, unsigned int n_actions, float learning_rate, float discount_factor, float exploration_rate, float exploration_decay, float exploration_min) {
    this->n_states = n_states;
    this->n_actions = n_actions;
    this->learning_rate = learning_rate;
    this->discount_factor = discount_factor;
    this->exploration_rate = exploration_rate;
    this->exploration_decay = exploration_decay;
    this->exploration_min = exploration_min;
    this->q_table = zeros({ n_states, n_actions });
}

void QLearning::choose_action(unsigned int state) {
//  # Exploit
//         if np.random.rand() > self.exploration_rate:
//             return np.argmax(self.q_table[state, :])
//         # Explore
//         return np.random.randint(0, self.n_actions)

    // Exploit
    if np.random.rand() > exploration_rate:
    // Explore

    // create
    np.random.rand()
    np.random.randint(
}

void QLearning::update(unsigned int state, unsigned int action, float reward, float next_state) {
    # Q-learning update rule
        best_next_action = np.argmax(self.q_table[next_state, :])
        q_target = reward + self.discount_factor * self.q_table[next_state, best_next_action]
        q_delta = q_target - self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * q_delta
        
        # Exploration rate decay
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay
}