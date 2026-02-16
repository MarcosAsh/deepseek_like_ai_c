#pragma once
#include <chrono>
#include <string>
#include <iostream>

// Simple RAII timer that logs the duration of a scope to stderr.
// Set Timer::enabled = true to activate output (off by default).
class Timer {
public:
    inline static bool enabled = false;

    explicit Timer(const std::string& name)
        : name_(name), start_(std::chrono::high_resolution_clock::now()) {}

    ~Timer() {
        if (!enabled) return;
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start_).count();
        std::cerr << "[TIMER] " << name_ << ": " << ms << " ms" << std::endl;
    }

private:
    std::string name_;
    std::chrono::high_resolution_clock::time_point start_;
};