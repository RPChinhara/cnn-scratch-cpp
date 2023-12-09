#pragma once

#include <string>
#include <vector>

class Environment
{
public:
    Environment() : actions({ "eat", "do_nothing", "up", "down", "left", "right" }), states({ "hungry", "neutral", "full" }) {
        num_states = states.size();
        num_actions = actions.size();
    }
    void Render();
    int Reset();
    std::tuple<int, int, bool> Step(const std::string& action);
    std::vector<std::string> actions;
    int num_states;
    size_t num_actions;
private:
    int CalculateReward();
    bool CheckTermination();
    std::vector<std::string> states;
    int current_state = std::distance(states.begin(), std::find(states.begin(), states.end(), "neutral"));
    size_t days_lived = 0;
    size_t days_without_eating = 0;
    size_t max_days = 50;
    size_t max_days_without_eating = 43;
};