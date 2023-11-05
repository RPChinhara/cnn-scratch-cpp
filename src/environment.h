#pragma once

#include <string>
#include <vector>

class Environment {
public:
    void render();
    int reset();
    std::tuple<int, int, bool> step(const std::string& action);
private:
    int calculate_reward();
    bool check_termination();
    std::vector<std::string> states  = { "hungry", "neutral", "full" };
    std::vector<std::string> actions = { "eat", "do_nothing" };
    int num_states                   = states.size(); 
    int num_actions                  = actions.size();
    int current_state                = std::distance(states.begin(), std::find(states.begin(), states.end(), "neutral")); // index of neutral
    int days_lived                   = 0;
    int days_without_eating          = 0;
    int max_days                     = 50;
    int max_days_without_eating      = 43;
};