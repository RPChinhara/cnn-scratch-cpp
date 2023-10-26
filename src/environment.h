#pragma once

#include <string>
#include <unordered_map>
#include <vector>

class Environment {
public:
    Environment();
    void render();
    std::unordered_map<std::string, std::string> reset();
    std::tuple<std::string, int, bool> step(const std::pair<int, char>& action);
private:
    int calculate_reward();
    bool check_termination();
    std::string update_thirstiness(int action);

    unsigned short days_lived;
    unsigned short thirsty_days;
    unsigned short max_days; // The desired number of days to live
    unsigned short max_thirsty_days;  // Number of consecutive days to tolerate thirstiness
    std::unordered_map<std::string, std::string> current_state;
};