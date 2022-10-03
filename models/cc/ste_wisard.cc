#include <vector>
#include <atomic>
#include <random>

#include "utils/utils.h"
#include "utils/mapping.h"
#include "utils/array.h"
#include "utils/atomic_operators.h"

#include "include/thread_pool.hpp"

using namespace std;

class BloomWiSARD55 {
public:

    int* tuple_lenghts;
    int** mappings;
    int num_rams;
    ArrayND<float>* rams;
    int num_keys;
    uint64_t* filter_keys;
    uint64_t filter_mask;
    int num_output;

    int num_threads;
    thread_pool* pool;
    vector<vector<int>> thread_rams; // Thread, Ram

    int batch_size;
    ArrayND<uint64_t>* batch_addrs;
    ArrayND<bool>* batch_addr_partial;


    
    BloomWiSARD55(int input_lenght, int tuple_lenght, int num_keys, int filter_tuple_lenght, int num_output, int batch_size) {
        
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
        this->rams = new ArrayND<float>({this->num_rams, (1 << filter_tuple_lenght), num_output});

        this->num_threads = std::thread::hardware_concurrency();        
        this->pool = new thread_pool(this->num_threads);
        this->pool->sleep_duration = 0;

        this->batch_size = batch_size;
        this->batch_addrs = new ArrayND<uint64_t>({batch_size, num_keys});
        this->batch_addr_partial = new ArrayND<bool>({batch_size, num_keys});

    };

    uint64_t get_addr(ArrayND<bool>& input, int i, int ram) {
        uint64_t addr = input(i, this->mappings[ram][0]) * this->filter_keys[0];
        for(int n = 1; n < this->tuple_lenghts[ram]; ++n)
            addr ^= input(i, this->mappings[ram][n]) * this->filter_keys[n];
        return addr;
    };

    void train(ArrayND<bool>& input, ArrayND<int>& classes) {
       
       this->pool->parallelize_loop(0, input.shape[0], [&input, &classes, this](int a, int b) {
            
            uint64_t keys;
            uint64_t addr;
            
            for(int i = a; i < b; ++i) {
                for(int r = 0; r < this->num_rams; ++r) {
                    keys = this->get_addr(input, i, r);
                    for(int k = 0; k < this->num_keys; ++k) {
                        addr = (keys >> (16 * k)) & this->filter_mask;
                        (*this->rams)(r , addr, classes[i]) = 1;
                    };
                };
            };

       });
    };

   ArrayND<atomic<int>>& predict(ArrayND<bool>& input, int begin=0, int end=-1) {
        
        if (end == -1) end = input.shape[0];
        ArrayND<atomic<int>> output({input.shape[0], this->num_output});
        
        this->pool->parallelize_loop(begin, end, [&input, &output, this](int a, int b) {

            uint64_t keys;
            uint64_t addr;
            bool output_and[this->num_output];

            for(int i = a; i < b; ++i) {
                for(int r = 0; r < this->num_rams; ++r) {
                    keys = this->get_addr(input, i, r);
                    for(int o = 0; o < this->num_output; ++o) {
                        addr = keys & this->filter_mask;
                        output_and[o] = (*this->rams)(r, addr, o);
                        for(int k = 1; k < this->num_keys; ++k) {
                            keys = keys >> 16;
                            addr = keys & this->filter_mask;
                            output_and[o] &= ((*this->rams)(r, addr, o) > 0);
                        };
                        output(i, o) += output_and[o];
                    };                   
                };
            };
        
        });
        
        return output;

    };

    // void predict(ArrayND<bool>& input, ArrayND<atomic<int>>& output, int begin=0, int end=-1) {
    //     if (end == -1) end = input.shape[0];
    //     this->run_parallel(BloomWiSARD55::predict_work, ref(input), ref(output), begin, end);
    // };

    ~BloomWiSARD55() {
        delete this->rams;
        delete this->pool;
        delete [] this->filter_keys;
        delete [] this->tuple_lenghts;
        for(int i = 0; i < this->num_rams; ++i)
            delete [] this->mappings[i];
        delete [] this->mappings;
    };

};