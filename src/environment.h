#pragma once

#include <string>
#include <vector>

class Environment {
public:
    Environment() : actions({ "eat", "do_nothing", "up", "down", "left", "right" }), states({ "hungry", "neutral", "full" }) {
        numStates = states.size();
        numActions = actions.size();
    }
    void Render();
    int Reset();
    std::tuple<int, int, bool> Step(const std::string& action);
    std::vector<std::string> actions;
    int numStates;
    int numActions;
private:
    int CalculateReward();
    bool CheckTermination();
    std::vector<std::string> states;
    int currentState = std::distance(states.begin(), std::find(states.begin(), states.end(), "neutral"));
    int daysLived = 0;
    int daysWithoutEating = 0;
    int maxDays = 50;
    int maxDaysWithoutEating = 43;
};