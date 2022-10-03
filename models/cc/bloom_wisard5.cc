#include <vector>
#include <atomic>
#include <random>

#include "utils/utils.h"
#include "utils/mapping.h"
#include "utils/array.h"
#include "utils/atomic_operators.h"

#include "include/thread_pool.hpp"

using namespace std;

class BloomWiSARD5 {
public:

    int* tuple_lenghts;
    int** mappings;
    int num_rams;
    ArrayND<bool>* rams;
    int num_keys;
    uint64_t* filter_keys;
    uint64_t filter_mask;
    int num_output;

    int num_threads;
    thread_pool* pool;
    vector<vector<int>> thread_rams; // Thread, Ram
    
    BloomWiSARD5(int input_lenght, int tuple_lenght, int num_keys, int filter_tuple_lenght, int num_output) {
        
        this->num_output = num_output;

        this->mappings = complete_mapping(input_lenght, tuple_lenght);
        int offset = input_lenght % tuple_lenght;
        this->num_rams = input_lenght / tuple_lenght + (offset > 0);
        this->tuple_lenghts = new int[this->num_rams];
        
        for(int i = 0; i < this->num_rams; ++i)
            this->tuple_lenghts[i] = ((i == this->num_rams - 1) && offset ? (offset) : (tuple_lenght));

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<uint64_t> distrib(0, (uint64_t)0 - 1);

        this->num_keys = num_keys;

        this->filter_keys = new uint64_t[tuple_lenght];
        for(int n = 0; n < tuple_lenght; ++n)
            this->filter_keys[n] = distrib(gen);

        this->filter_mask = (1 << filter_tuple_lenght) - 1;
        this->rams = new ArrayND<bool>({this->num_rams, (1 << filter_tuple_lenght), num_output});

        this->num_threads = std::thread::hardware_concurrency();        
        this->pool = new thread_pool(this->num_threads);

        for(int i = 0; i < this->num_threads; ++i)
            this->thread_rams.push_back(vector<int>());

        int thread = 0;
        
        for(int i = 0; i < this->num_rams; ++i) {
            this->thread_rams[thread].push_back(i);
            thread = (thread + 1) % this->num_threads;     
        };   

    };

    uint64_t get_addr(ArrayND<bool>& input, int i, int ram) {
        uint64_t addr = input(i, this->mappings[ram][0]) * this->filter_keys[0];
        for(int n = 1; n < this->tuple_lenghts[ram]; ++n)
            addr ^= input(i, this->mappings[ram][n]) * this->filter_keys[n];
        return addr;
    };

    template <typename F, typename... A>
    void run_parallel(const F &task, const A &...args) {
        for(int i = 0; i < this->num_threads; ++i)
            this->pool->push_task(std::bind(task, i, this, args...));
        this->pool->wait_for_tasks();
    };   

    void train_function(int thread, ArrayND<bool>& input, ArrayND<int>& classes) {
        int ram;
        uint64_t keys;
        uint64_t addr;
        for(int i = 0; i < input.shape[0]; ++i) {
            for(size_t j = 0; j < this->thread_rams[thread].size(); ++j) {
                ram = this->thread_rams[thread][j];
                keys = this->get_addr(input, i, ram);
                for(int k = 0; k < this->num_keys; ++k) {
                    addr = (keys >> (16 * k)) & this->filter_mask;
                    (*this->rams)(ram , addr, classes[i]) = 1;
                };
            };
        };
    };

    static void train_work(int thread, BloomWiSARD5* obj, ArrayND<bool>& input, ArrayND<int>& classes) {
        obj->train_function(thread, input, classes);
    };

    void train(ArrayND<bool>& input, ArrayND<int>& classes) {
        this->run_parallel(BloomWiSARD5::train_work, ref(input), ref(classes));
    };

    void predict_function(int thread, ArrayND<bool>& input, ArrayND<atomic<int>>& output, int begin, int end) {
        int ram;
        uint64_t keys;
        uint64_t addr;
        bool output_and[this->num_output];
        for(int i = begin; i < end; ++i) {
            for(size_t j = 0; j < this->thread_rams[thread].size(); ++j) {
                ram = this->thread_rams[thread][j];
                keys = this->get_addr(input, i, ram);
                for(int o = 0; o < this->num_output; ++o)
                    output_and[o] = 1;
                for(int k = 0; k < this->num_keys; ++k) {
                    addr = (keys >> (16 * k)) & this->filter_mask;
                    for(int o = 0; o < this->num_output; ++o)
                        output_and[o] &= (*this->rams)(ram, addr, o);
                };
                for(int o = 0; o < this->num_output; ++o)
                    output(i, o) += output_and[o];
            };
        };
    };

    static void predict_work(int thread, BloomWiSARD5* obj, ArrayND<bool>& input, ArrayND<atomic<int>>& output, int begin, int end) {
        obj->predict_function(thread, input, output, begin, end);
    };

    ArrayND<atomic<int>> predict(ArrayND<bool>& input, int begin=0, int end=-1) {
        if (end == -1) end = input.shape[0];
        ArrayND<atomic<int>> output({input.shape[0], this->num_output});
        this->run_parallel(BloomWiSARD5::predict_work, ref(input), ref(output), begin, end);
        return output;
    };

    void predict(ArrayND<bool>& input, ArrayND<atomic<int>>& output, int begin=0, int end=-1) {
        if (end == -1) end = input.shape[0];
        this->run_parallel(BloomWiSARD5::predict_work, ref(input), ref(output), begin, end);
    };

    ~BloomWiSARD5() {
        delete this->rams;
        delete this->pool;
        delete [] this->filter_keys;
        delete [] this->tuple_lenghts;
        for(int i = 0; i < this->num_rams; ++i)
            delete [] this->mappings[i];
        delete [] this->mappings;
    };

};