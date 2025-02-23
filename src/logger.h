#pragma once

#include <fstream>
#include <iostream>

class logger {
public:
    static void init();
    static void log(const std::string& message);
    static void close();

private:
    static std::ofstream log_file;
};