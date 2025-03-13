#pragma once

#include <chrono>

class CPUTimer
{
public:
    CPUTimer();
    void start();
    void stop();
    double getElapsedTime();
    double getElapsedTimeMs();
    void reset();

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> m_startTime;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_endTime;
    bool m_isRunning;
};