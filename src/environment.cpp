#include "environment.h"

#include <iostream>
#include <windows.h>

Environment::Environment(const std::string& secret_word) {
    this->secret_word = secret_word;
    current_guess.insert(current_guess.end(), secret_word.length(), '_');
    max_attempts = secret_word.length() * 2;
    attempts_made = 0;
    done = false;
}

void Environment::render() {
    std::cout << "Current guess: " << insert_space(current_guess) << std::endl;
    std::cout << "Attempts made: " << this->attempts_made << std::endl;
}

std::string Environment::reset() {
    current_guess.clear();
    current_guess.insert(current_guess.end(), secret_word.length(), '_');
    attempts_made = 0;
    done = false;
    return insert_space(current_guess);
}

std::tuple<std::string, int, bool> Environment::step(const std::pair<int, char>& action) {
    if (done) {
        MessageBox(NULL, "The game is over. Please reset the environment.", "Error", MB_ICONERROR | MB_OK);
        ExitProcess(1);
    }

    // Unpack the action, which is expected to be a tuple of (position, letter)
    int position = action.first;
    char letter  = action.second;

    if (secret_word[position] == letter)
        current_guess[position] = letter;

    attempts_made += 1;

    if (attempts_made >= max_attempts) {
        done = true;
        return std::make_tuple(insert_space(current_guess), -1, done); // -1 reward for failure
    }
    
    if (current_guess == secret_word) {
        done = true;
        return std::make_tuple(insert_space(current_guess), 1, done); // 1 reward for failure
    }

    return std::make_tuple(insert_space(current_guess), 0, done); // 0 reward for failure
}

std::string Environment::insert_space(std::string& str) {
    std::string spaced_string;
    for (size_t i = 0; i < str.length(); ++i) {
        spaced_string += str[i];
        // Add a space after each character, except for the last character
        if (i < str.length() - 1) {
            spaced_string += ' ';
        }
    }
    return spaced_string;
}