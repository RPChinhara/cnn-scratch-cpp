#pragma once

#include <string>
#include <vector>

class Environment
{
public:
    Environment() : actions({ "do_nothing", "up", "down", "left", "right" }), states({ "hungry", "neutral", "full" }) {
        num_states = states.size();
        num_actions = actions.size();
    }
    void Render();
    size_t Reset();
    std::tuple<size_t, int, bool> Step(const std::string& action);
    std::vector<std::string> actions;
    size_t num_states;
    size_t num_actions;
private:
    int CalculateReward();
    bool CheckTermination();
    std::vector<std::string> states;
    size_t current_state = std::distance(states.begin(), std::find(states.begin(), states.end(), "neutral"));
    std::string current_action;
    size_t days_lived = 0;
    size_t max_days = 50;
    size_t days_without_eating = 0;
    size_t days_without_drinking = 0;
    size_t max_days_without_eating = 43;
    size_t max_days_without_drinking = 3;
};