#ifndef MY_THREAD_UTILS_H
#define MY_THREAD_UTILS_H

#include <atomic>


static std::atomic<int>& operator+= (std::atomic<int>& atomicNumber, int increment) {
    
    int oldValue;
    int newValue;
    
    do {
        oldValue = atomicNumber.load(std::memory_order_relaxed);
        newValue = oldValue + increment;
    } while (!atomicNumber.compare_exchange_weak(oldValue, newValue, std::memory_order_release, std::memory_order_relaxed));
    
    return atomicNumber;
};

static std::atomic<float>& operator+= (std::atomic<float>& atomicNumber, float increment) {
    
    float oldValue;
    float newValue;
    
    do {
        oldValue = atomicNumber.load(std::memory_order_relaxed);
        newValue = oldValue + increment;
    } while (!atomicNumber.compare_exchange_weak(oldValue, newValue, std::memory_order_release, std::memory_order_relaxed));
    
    return atomicNumber;
};


#endif