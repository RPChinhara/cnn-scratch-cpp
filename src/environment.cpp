#include "environment.h"

#include <iostream>

Environment::Environment(const std::string& secret_word) {
    this->secret_word = secret_word;
    current_guess.insert(current_guess.end(), secret_word.length(), '_');
    max_attempts = secret_word.length() * 2;
    attempts_made = 0;
    done = false;
}

void Environment::render() {
    std::cout << "Current guess: " << current_guess << std::endl; // TODO: not done yet
    std::cout << "Attempts made: " << this->attempts_made << std::endl;
}

std::string Environment::reset() {
    current_guess.insert(current_guess.end(), secret_word.length(), '_');
    attempts_made = 0;
    done = false;
    return current_guess;
}

std::tuple<std::string, int, bool> Environment::step(const std::pair<int, char>& action) {
    if (done)
        std::cerr << "The game is over. Please reset the environment." << std::endl;

    // Unpack the action, which is expected to be a tuple of (position, letter)
    int position = action.first;
    char letter  = action.second;

    if (secret_word[position] == letter)
        current_guess[position] = letter;

    attempts_made += 1;

    if (attempts_made >= max_attempts) {
        done = true;
        return std::make_tuple(current_guess, -1, done); // -1 reward for failure
    }
    
    if (current_guess == secret_word) {
        done = true;
        return std::make_tuple(current_guess, 1, done); // 1 reward for failure
    }

    return std::make_tuple(current_guess, 0, done); // 0 reward for failure
}