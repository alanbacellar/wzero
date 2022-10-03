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
    int filter_tuple_lenght;
    uint64_t* filter_keys;
    uint64_t key_mask;
    uint64_t* ram_keys_masks;
    int num_output;

    int num_threads;
    thread_pool* pool;
    vector<vector<vector<int>>> thread_simd_rams; // Thread, SIMD, Ram
    
    BloomWiSARD5(int input_lenght, int tuple_lenght, int num_keys, int filter_tuple_lenght, int num_output) {
        
        this->num_output = num_output;

        this->mappings = complete_mapping(input_lenght, tuple_lenght);
        int offset = input_lenght % tuple_lenght;
        this->num_rams = input_lenght / tuple_lenght + (offset > 0);
        this->tuple_lenghts = new int[this->num_rams];
        
        for(int i = 0; i < this->num_rams; ++i)
            this->tuple_lenghts[i] = ((i == this->num_rams - 1) && offset ? (offset) : (tuple_lenght));

        this->num_keys = num_keys;
        this->filter_tuple_lenght = filter_tuple_lenght;

        this->key_mask = ((uint64_t)1 << filter_tuple_lenght) - 1;
        uint64_t keys_mask = ((uint64_t)1 << (filter_tuple_lenght * num_keys)) - 1;

        this->rams = new ArrayND<bool>({this->num_rams, (1 << filter_tuple_lenght), num_output});

        this->num_threads = std::thread::hardware_concurrency();        
        this->pool = new thread_pool(this->num_threads);


        int rams_per_thread = this->num_rams / this->num_threads;
        int rams_per_thread_offset = this->num_rams % this->num_threads;

        int rams_per_simd = 64 / (num_keys * filter_tuple_lenght);

        this->ram_keys_masks = new uint64_t[rams_per_simd];
        for(int r = 0; r < rams_per_simd; ++r)
            this->ram_keys_masks[r] = keys_mask << this->filter_tuple_lenght * this->num_keys * r;

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<uint64_t> distrib(0, keys_mask);

        uint64_t key;

        this->filter_keys = new uint64_t[tuple_lenght];
        for(int n = 0; n < tuple_lenght; ++n) {
            this->filter_keys[n] = distrib(gen);
            for(int r = 0; r < rams_per_simd; ++r)
                this->filter_keys[n] |= this->filter_keys[n] << (filter_tuple_lenght * num_keys * r);
        };


        int ram = 0;

        for(int i = 0; i < this->num_threads; ++i) {
            this->thread_simd_rams.push_back(vector<vector<int>>());
            int num_rams = rams_per_thread + (i < rams_per_thread_offset);
            int simd_offset = num_rams % rams_per_simd;
            int num_simd = num_rams / rams_per_simd + (simd_offset > 0);
            for(int j = 0; j < num_simd; ++j) {
                this->thread_simd_rams[i].push_back(vector<int>());
                int simd_num_rams = ((j == num_simd - 1) && simd_offset ? (simd_offset) : (rams_per_simd));
                for(int k = 0; k < simd_num_rams; ++k) {
                    this->thread_simd_rams[i][j].push_back(ram);
                    ++ram; 
                };
            };
        };
            


    };

    uint64_t get_addr(ArrayND<bool>& input, int i, int* ram, int num_rams) {
        
        uint64_t keys = 0;
        uint64_t keys_tmp;

        for(int r = 0; r < num_rams; ++r)
            keys |= input(i, this->mappings[ram[r]][0]) * this->ram_keys_masks[r];
        keys &= this->filter_keys[0];

        for(int n = 1; n < this->tuple_lenghts[ram]; ++n) {
            keys_tmp = 0;
            for(size_t r = 0; r < this->thread_simd_rams[thread][s].size(); ++r) {
                ram = this->thread_simd_rams[thread][s][r];
                keys_tmp |= input(i, this->mappings[ram][n]) * this->ram_keys_masks[r];
            };
            keys ^= keys_tmp & this->filter_keys[n];
        };
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
        uint64_t keys_tmp;
        uint64_t addr;
        uint64_t mask;

        for(int i = 0; i < input.shape[0]; ++i) {
            for(size_t s = 0; s < this->thread_simd_rams[thread].size(); ++s) {
                
                keys = 0;

                for(size_t r = 0; r < this->thread_simd_rams[thread][s].size(); ++r) {
                    ram = this->thread_simd_rams[thread][s][r];
                    keys |= input(i, this->mappings[ram][0]) * this->ram_keys_masks[r];
                };
                keys &= this->filter_keys[0];

                for(int n = 1; n < this->tuple_lenghts[ram]; ++n) {
                    keys_tmp = 0;
                    for(size_t r = 0; r < this->thread_simd_rams[thread][s].size(); ++r) {
                        ram = this->thread_simd_rams[thread][s][r];
                        keys_tmp |= input(i, this->mappings[ram][n]) * this->ram_keys_masks[r];
                    };
                    keys ^= keys_tmp & this->filter_keys[n];
                };

                for(size_t r = 0; r < this->thread_simd_rams[thread][j].size(); ++r) {
                    
                    keys |= input(i, this->mappings[ram][0]) * this->filter_keys[0];
                    for(int n = 1; n < this->tuple_lenghts[ram]; ++n)
                        keys ^= input(i, this->mappings[ram][n]) * this->filter_keys[n];
                    
                    int ram = this->thread_simd_rams[thread][j][r];
                    
                    for(int k = 0; k < this->num_keys; ++k) {
                        addr = (keys >> (16 * k)) & this->filter_mask;
                        (*this->rams)(ram , addr, classes[i]) = 1;
                    };
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
            for(size_t j = 0; j < this->thread_simd_rams[thread].size(); ++j) {
                ram = this->thread_simd_rams[thread][j];
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