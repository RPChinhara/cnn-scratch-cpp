#include "environment.h"

#include <iostream>
#include <windows.h>

Environment::Environment() {
    days_lived                   = 0;
    thirsty_days                 = 0;
    max_days                     = 50;
    max_thirsty_days             = 3;
    current_state["thirstiness"] = "neutral";
}

void Environment::render() {
    std::cout << "Current guess: " << std::endl;
    std::cout << "Attempts made: " << std::endl;
}

std::unordered_map<std::string, std::string> Environment::reset() {
    days_lived                   = 0;
    thirsty_days                 = 0;
    current_state["thirstiness"] = "neutral";

    return current_state;
}

std::tuple<std::string, int, bool> Environment::step(const std::pair<int, char>& action) {
    if (true) {
        MessageBox(NULL, "The game is over. Please reset the environment.", "Error", MB_ICONERROR | MB_OK);
        ExitProcess(1);
    }

    bool done = false;

    return std::make_tuple("cat", 0, done); // 0 reward for failure
}

void Environment::update_thirstiness() {

}