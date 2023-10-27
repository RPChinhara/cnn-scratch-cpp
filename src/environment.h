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

    int num_states; // Total number of states (5 * 5 * 5)
    int num_actions; // Total number of action combinations (3^4)
    int days_lived;
    int thirsty_days;
    int max_days; // The desired number of days to live
    int max_thirsty_days;  // Number of consecutive days to tolerate thirstiness
    States current_state;
};