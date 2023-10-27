#pragma once

#include <string>
#include <unordered_map>
#include <vector>

using States = std::unordered_map<std::string, int>;

class Environment {
public:
    Environment();
    void render();
    States reset();
    std::tuple<States, int, bool> step(int action);
private:
    int calculate_reward();
    bool check_termination();
    int update_thirstiness(int action);

    unsigned short days_lived;
    unsigned short thirsty_days;
    unsigned short max_days; // The desired number of days to live
    unsigned short max_thirsty_days;  // Number of consecutive days to tolerate thirstiness
    States current_state;
};