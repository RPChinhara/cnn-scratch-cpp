#include <windows.h>

#include "logger.h"
#include "tensor.h"

std::ofstream logger::log_file;

void logger::init() {
    AllocConsole();
    freopen_s((FILE**)stdout, "CONOUT$", "w", stdout);
    freopen_s((FILE**)stderr, "CONOUT$", "w", stderr);

    // log_file.open("debug_log.txt");
}

void logger::log(const tensor& t) {
    std::cout << t << std::endl;

    // if (log_file.is_open())
    //     log_file << message << std::endl;
}

void logger::log(const std::string& message) {
    std::cout << message << std::endl;

    // if (log_file.is_open())
    //     log_file << message << std::endl;
}

void logger::close() {
    // if (log_file.is_open())
    //     log_file.close();
}
