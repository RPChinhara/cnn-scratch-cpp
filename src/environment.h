#pragma once

#include <string>
#include <vector>

class Environment {
public:
    Environment() : actions({ "eat", "do_nothing", "up", "down", "left", "right" }), states({ "hungry", "neutral", "full" }) {
        num_states = states.size();
        num_actions = actions.size();
    }
    void render();
    int reset();
    std::tuple<int, int, bool> step(const std::string& action);
    std::vector<std::string> actions;
    int num_states;
    int num_actions;
private:
    int calculate_reward();
    bool check_termination();
    std::vector<std::string> states;
    int current_state = std::distance(states.begin(), std::find(states.begin(), states.end(), "neutral")); // Index of neutral
    int days_lived = 0;
    int days_without_eating = 0;
    int max_days = 50;
    int max_days_without_eating = 43;
};