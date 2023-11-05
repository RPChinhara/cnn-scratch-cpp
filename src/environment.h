#pragma once

#include <string>
#include <unordered_map>
#include <vector>

using States = std::unordered_map<std::string, int>;

class Environment {
public:
    void render();
    int reset();
    std::tuple<States, int, bool> step(int action);
private:
    int calculate_reward();
    bool check_termination();
    int update_thirstiness(int action);

    std::vector<std::string> states  = { "hungry", "neutral", "full" };
    std::vector<std::string> actions = { "eat", "do_nothing" };
    int num_states                   = states.size(); 
    int num_actions                  = actions.size();
    int current_state                = 1; // corresponds to "neutral"
    int days_lived                   = 0;
    int days_without_eating          = 0;
    int max_days                     = 50;
    int max_days_without_eating      = 43;
};