#include <unordered_map>
#include <vector>
#include <atomic>
#include <random>

#include "utils/utils.h"
#include "utils/mapping.h"
#include "utils/array.h"
#include "utils/atomic_operators.h"

#include "include/thread_pool.hpp"
#include "include/flat_hash_map.hpp"

using namespace std;

class BloomRam2 {
public:

    int tuple_lenght;
    int* mapping;
    int num_keys;
    int filter_tuple_lenght;
    int num_output;
    bool* output;
    uint32_t* keys_addr;

    ArrayND<uint32_t>* filter_keys;
    ArrayND<bool>* memory;

    BloomRam2(int tuple_lenght, int* mapping, int num_keys, int filter_tuple_lenght, int num_output) {
        this->tuple_lenght = tuple_lenght;
        this->mapping = mapping;
        this->num_keys = num_keys;
        this->filter_tuple_lenght = filter_tuple_lenght;
        this->num_output = num_output;
        this->output = new bool[num_output];
        this->keys_addr = new uint32_t[num_keys];

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<uint32_t> distrib(0, (1 << filter_tuple_lenght) - 1);

        this->filter_keys = new ArrayND<uint32_t>({num_keys, tuple_lenght});
        for(int k = 0; k < num_keys; ++k) {
            for(int n = 0; n < this->tuple_lenght; ++n) {
                (*this->filter_keys)(k, n) = distrib(gen);
            };
        };

        this->memory = new ArrayND<bool>({(1 << filter_tuple_lenght), num_output});
    }; 

    uint32_t get_addr(ArrayND<bool>& input, int i, int k) {
        uint32_t addr = input(i, this->mapping[0]) * (*this->filter_keys)(k, 0);
        for(int n = 1; n < this->tuple_lenght; ++n) {
            addr ^= input(i, this->mapping[n]) * (*this->filter_keys)(k, n);
        };
        return addr;
    };

    void train(ArrayND<bool>& input, int i, int class_) {
        for(int k = 0; k < this->num_keys; ++k)
            (*this->memory)(this->get_addr(input, i, k), class_) = 1;
    };

    void predict(ArrayND<bool>& input, ArrayND<atomic<int>>& output, int i) {
        uint32_t addr;
        for(int o = 0; o < this->num_output; ++o)
            this->output[o] = 1;
        for(int k = 0; k < this->num_keys; ++k) {
            addr = this->get_addr(input, i, k);
            for(int o = 0; o < this->num_output; ++o)
                this->output[o] &= (*this->memory)(addr, o);
        };
        for(int o = 0; o < this->num_output; ++o)
            output(i, o) += this->output[o];
    };

    ~BloomRam2() {
        
    };

};


class BloomWiSARD2 {
public:

    int input_lenght;
    int tuple_lenght;
    int num_classes;

    int num_rams;
    BloomRam2** rams;

    int num_threads;
    thread_pool* pool;
    vector<vector<int>> thread_rams; // Thread, Ram
    
    BloomWiSARD2(int input_lenght, int tuple_lenght, int num_keys, int filter_tuple_lenght, int num_classes) {
        
        this->input_lenght = input_lenght;
        this->tuple_lenght = tuple_lenght;
        this->num_classes = num_classes;

        int** mapping = complete_mapping(input_lenght, tuple_lenght);
        int offset = input_lenght % tuple_lenght;
        this->num_rams = input_lenght / tuple_lenght + (offset > 0);
        this->rams = new BloomRam2*[this->num_rams];
        
        for(int i = 0; i < this->num_rams; ++i) {
            int ram_tuple_lenght = ((i == this->num_rams - 1) && offset ? (offset) : (tuple_lenght));
            this->rams[i] = new BloomRam2(ram_tuple_lenght, mapping[i], num_keys, filter_tuple_lenght, num_classes);
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

    static void train_work(int thread, BloomWiSARD2* obj, ArrayND<bool>& input, ArrayND<int>& classes) {
        int ram;
        for(int i = 0; i < input.shape[0]; ++i) {
            for(size_t j = 0; j < obj->thread_rams[thread].size(); ++j) {
                ram = obj->thread_rams[thread][j];
                obj->rams[ram]->train(input, i, classes[i]);
            };
        };
    };

    void train(ArrayND<bool>& input, ArrayND<int>& classes) {
        this->run_parallel(BloomWiSARD2::train_work, ref(input), ref(classes));
    };

    static void predict_work(int thread, BloomWiSARD2* obj, ArrayND<bool>& input, ArrayND<atomic<int>>& output, int begin, int end) {
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
        this->run_parallel(BloomWiSARD2::predict_work, ref(input), ref(output), begin, end);
        return output;
    };

    void predict(ArrayND<bool>& input, ArrayND<atomic<int>>& output, int begin=0, int end=-1) {
        if (end == -1) end = input.shape[0];
        this->run_parallel(BloomWiSARD2::predict_work, ref(input), ref(output), begin, end);
    };

    ~BloomWiSARD2() {
        for(int i = 0; i < this->num_rams; ++i) {
            delete this->rams[i];
        };
        delete [] this->rams;
        delete this->pool;
    };

};