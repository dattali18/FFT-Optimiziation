#include "cpu_timer.h"

CPUTimer::CPUTimer()
    : m_isRunning(false)
{
}

void CPUTimer::start()
{
    m_startTime = std::chrono::high_resolution_clock::now();
    m_isRunning = true;
}

void CPUTimer::stop()
{
    m_endTime = std::chrono::high_resolution_clock::now();
    m_isRunning = false;
}

double CPUTimer::getElapsedTime()
{
    if (m_isRunning)
    {
        stop();
    }

    std::chrono::duration<double> elapsed = m_endTime - m_startTime;
    return elapsed.count();
}

double CPUTimer::getElapsedTimeMs()
{
    return getElapsedTime() * 1000.0;
}