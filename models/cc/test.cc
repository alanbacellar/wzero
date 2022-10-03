#include <iostream>
#include <unordered_map>
#include "include/thread_pool.hpp"
#include "utils/array.h"
#include <atomic>
#include "utils/atomic_operators.h"
#include <random>
#include <bitset>

using namespace std;

class Bitset {
public:

    long* data;
    int num_bits;
    int word_size;

    Bitset(int num_bits) {
        this->num_bits = num_bits;
        this->word_size = 8*sizeof(long);
    };

    Bitset(ArrayND<bool>& array) {
        this->num_bits = array.size;
        this->word_size = 8*sizeof(long);
        this->data = new long[this->num_bits / this->word_size + ((this->num_bits % this->word_size) > 0)]();
        for(int i = 0; i < array.size; ++i) {
            int block = i / this->word_size;
            int index = i % this->word_size;
            this->data[block] |= (long)array(i) << index;
        };
    };

    bool operator[](int i) {
        int block = i / this->word_size;
        int index = i % this->word_size;
        return (this->data[block] & ((long)(1) << index));// >> index;
    };

    ~Bitset() {

    };


};

struct Timer
{
    Timer() { clock_gettime(CLOCK_MONOTONIC, &from); }
    struct timespec from;
    double elapsed() const
    {
        struct timespec to;
        clock_gettime(CLOCK_MONOTONIC, &to);
        return to.tv_sec - from.tv_sec + 1E-9 * (to.tv_nsec - from.tv_nsec);
    }
};

int main() {

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distrib(0, 1);

    int n = 10000000;

    cout << "a" << endl;
    ArrayND<bool> a({n});

    for(int i = 0; i < n; ++i)
        a(i) = distrib(gen);

    cout << "b" << endl;
    Bitset b(a);

    cout << "s" << endl;

    Timer t1;
    int sa = 0;
    for(int i = 0; i < n; ++i) 
        sa += a(i);
    double t1e = t1.elapsed();
    cout << sa << endl;
    cout << t1e << endl;
    
    Timer t2;
    int sb = 0;
    for(int i = 0; i < n; ++i) 
        sb += b[i];
    double t2e = t2.elapsed();
    cout << sb << endl;
    cout << t2e << endl;

    cout << "c" << endl;

    std::bitset<10000000> c;
    for(int i = 0; i < n; ++i)
        c.set(i, a(i));
    
    cout << "s2" << endl;
    
    Timer t3;
    int sc = 0;
    for(int i = 0; i < n; ++i) 
        sc += c[i];
    double t3e = t3.elapsed();
    cout << sc << endl;
    cout << t3e << endl;

    return 0;
};