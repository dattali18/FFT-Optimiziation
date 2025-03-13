#pragma once

#include "cpu_timer.h"

#define MIN_TIME(code)                                               \
    {                                                                \
        CPUTimer timer;                                              \
        double min_time = std::numeric_limits<double>::max();        \
        for (int i = 0; i < 100; i++)                                \
        {                                                            \
            timer.start();                                           \
            code;                                                    \
            timer.stop();                                            \
            min_time = std::min(min_time, timer.getElapsedTimeMs()); \
        }                                                            \
        std::cout << "Min time: " << min_time << " ms" << std::endl; \
    }