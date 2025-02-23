#pragma once

#include <fstream>
#include <iostream>

class tensor;

class logger {
public:
    static void init();
    static void log(const tensor& t);
    static void log(const std::string& message);
    static void close();

private:
    static std::ofstream log_file;
};