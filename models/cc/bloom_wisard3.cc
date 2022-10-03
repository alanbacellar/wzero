#include <vector>
#include <atomic>
#include <random>

#include "utils/utils.h"
#include "utils/mapping.h"
#include "utils/array.h"
#include "utils/atomic_operators.h"

#include "include/thread_pool.hpp"

using namespace std;

class BloomRam3 {
public:

    int tuple_lenght;
    int* mapping;
    int num_keys;
    int filter_tuple_lenght;
    int filter_mask;
    int num_output;
    bool* output;

    uint64_t* filter_keys;
    ArrayND<bool>* memory;

    BloomRam3(int tuple_lenght, int* mapping, int num_keys, int filter_tuple_lenght, int num_output) {
        this->tuple_lenght = tuple_lenght;
        this->mapping = mapping;
        this->num_keys = num_keys;
        this->filter_tuple_lenght = filter_tuple_lenght;
        this->num_output = num_output;
        this->output = new bool[num_output];

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<uint64_t> distrib(0, (uint64_t)-1);

        this->filter_keys = new uint64_t[tuple_lenght];
        for(int n = 0; n < this->tuple_lenght; ++n)
            this->filter_keys[n] = distrib(gen);

        this->filter_mask = (1 << filter_tuple_lenght) - 1;
        this->memory = new ArrayND<bool>({(1 << filter_tuple_lenght), num_output});
    }; 

    uint64_t get_addr(ArrayND<bool>& input, int i) {
        uint64_t addr = input(i, this->mapping[0]) * this->filter_keys[0];
        for(int n = 1; n < this->tuple_lenght; ++n)
            addr ^= input(i, this->mapping[n]) * this->filter_keys[n];
        return addr;
    };

    void train(ArrayND<bool>& input, int i, int class_) {
        uint64_t keys = this->get_addr(input, i);
        uint64_t addr;
        for(int k = 0; k < this->num_keys; ++k) {
            addr = (keys >> (16 * k)) & this->filter_mask;
            (*this->memory)(addr, class_) = 1;
        };
    };

    void predict(ArrayND<bool>& input, ArrayND<atomic<int>>& output, int i) {
        uint64_t keys = this->get_addr(input, i);
        uint64_t addr;
        for(int o = 0; o < this->num_output; ++o)
            this->output[o] = 1;
        for(int k = 0; k < this->num_keys; ++k) {
            addr = (keys >> (16 * k)) & this->filter_mask;
            for(int o = 0; o < this->num_output; ++o)
                this->output[o] &= (*this->memory)(addr, o);
        };
        for(int o = 0; o < this->num_output; ++o)
            output(i, o) += this->output[o];
    };

    ~BloomRam3() {
        
    };

};


class BloomWiSARD3 {
public:

    int input_lenght;
    int tuple_lenght;
    int num_classes;

    int num_rams;
    BloomRam3** rams;

    int num_threads;
    thread_pool* pool;
    vector<vector<int>> thread_rams; // Thread, Ram
    
    BloomWiSARD3(int input_lenght, int tuple_lenght, int num_keys, int filter_tuple_lenght, int num_classes) {
        
        this->input_lenght = input_lenght;
        this->tuple_lenght = tuple_lenght;
        this->num_classes = num_classes;

        int** mapping = complete_mapping(input_lenght, tuple_lenght);
        int offset = input_lenght % tuple_lenght;
        this->num_rams = input_lenght / tuple_lenght + (offset > 0);
        this->rams = new BloomRam3*[this->num_rams];
        
        for(int i = 0; i < this->num_rams; ++i) {
            int ram_tuple_lenght = ((i == this->num_rams - 1) && offset ? (offset) : (tuple_lenght));
            this->rams[i] = new BloomRam3(ram_tuple_lenght, mapping[i], num_keys, filter_tuple_lenght, num_classes);
        };

        delete mapping;

        this->num_threads = std::thread::hardware_concurrency();        
        this->pool = new thread_pool(this->num_threads);

        for(int i = 0; i < this->num_threads; ++i) {
            this->thread_rams.push_back(vector<int>());
        };

        int thread = 0;
        
        for(int i = 0; i < this->num_rams; ++i) {
            this->thread_rams[thread].push_back(i);
            thread = (thread + 1) % this->num_threads;     
        };   

    };

    template <typename F, typename... A>
    void run_parallel(const F &task, const A &...args) {
        for(int i = 0; i < this->num_threads; ++i)
            this->pool->push_task(std::bind(task, i, this, args...));
        this->pool->wait_for_tasks();
    };   

    static void train_work(int thread, BloomWiSARD3* obj, ArrayND<bool>& input, ArrayND<int>& classes) {
        int ram;
        for(int i = 0; i < input.shape[0]; ++i) {
            for(size_t j = 0; j < obj->thread_rams[thread].size(); ++j) {
                ram = obj->thread_rams[thread][j];
                obj->rams[ram]->train(input, i, classes[i]);
            };
        };
    };

    void train(ArrayND<bool>& input, ArrayND<int>& classes) {
        this->run_parallel(BloomWiSARD3::train_work, ref(input), ref(classes));
    };

    static void predict_work(int thread, BloomWiSARD3* obj, ArrayND<bool>& input, ArrayND<atomic<int>>& output, int begin, int end) {
        int ram;
        for(int i = begin; i < end; ++i) {
            for(size_t j = 0; j < obj->thread_rams[thread].size(); ++j) {
                ram = obj->thread_rams[thread][j];
                obj->rams[ram]->predict(input, output, i);
            };
        };
    };

    ArrayND<atomic<int>> predict(ArrayND<bool>& input, int begin=0, int end=-1) {
        if (end == -1) end = input.shape[0];
        ArrayND<atomic<int>> output({input.shape[0], this->num_classes});
        this->run_parallel(BloomWiSARD3::predict_work, ref(input), ref(output), begin, end);
        return output;
    };

    void predict(ArrayND<bool>& input, ArrayND<atomic<int>>& output, int begin=0, int end=-1) {
        if (end == -1) end = input.shape[0];
        this->run_parallel(BloomWiSARD3::predict_work, ref(input), ref(output), begin, end);
    };

    ~BloomWiSARD3() {
        for(int i = 0; i < this->num_rams; ++i) {
            delete this->rams[i];
        };
        delete [] this->rams;
        delete this->pool;
    };

};