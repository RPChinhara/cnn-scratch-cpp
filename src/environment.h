#pragma once

#include <string>
#include <vector>

class Environment {
public:
    Environment(const std::string& secret_word);
    void render();
    std::string reset();
    std::tuple<std::string, int, bool> step(const std::pair<int, char>& action);
private:
    std::string insert_space(std::string& str);
    
    std::string secret_word;
    std::string current_guess;
    int max_attempts;
    unsigned int attempts_made;
    bool done;
};