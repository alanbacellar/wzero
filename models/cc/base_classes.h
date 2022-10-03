#ifndef BASE_CLASSES_H
#define BASE_CLASSES_H

#include <unordered_map>
#include <vector>
#include <atomic>

#include "utils/utils.h"
#include "utils/mapping.h"
#include "utils/array.h"
#include "utils/atomic_operators.h"

#include "include/thread_pool.hpp"
#include "include/flat_hash_map.hpp"

using namespace std;

template <class KEY_TYPE, class VALUE_TYPE, class OUTPUT_VALUE_TYPE>
class RamBase {
public:

    int input_lenght;
    int tuple_lenght;
    int* mapping;
    int class_;
    
    KEY_TYPE num;

    //unordered_map<KEY_TYPE, VALUE_TYPE> memory;
    ska::flat_hash_map<KEY_TYPE, VALUE_TYPE> memory;

    RamBase(int input_lenght, int tuple_lenght, int* mapping, int class_) {
        this->input_lenght = input_lenght;
        this->tuple_lenght = tuple_lenght;
        this->mapping = mapping;
        this->class_ = class_;
    }; 

    KEY_TYPE get_addr(ArrayND<bool>& input, int i) {
        KEY_TYPE addr = 0;
        for(int j = 0; j < this->tuple_lenght; ++j) { 
            // addr |= input(i, this->mapping[j]) << j;
            addr |= input[input.strides[0] * i + this->mapping[j]] << j;
        };
        return addr;
    };

    virtual void train(ArrayND<bool>& input, int i) {
        this->memory[this->get_addr(input, i)] = 1;
    };

    virtual void predict(ArrayND<bool>& input, ArrayND<atomic<OUTPUT_VALUE_TYPE>>& output, int i) {
        if (this->memory.count(this->get_addr(input, i))) {
            // output(i, this->class_) += 1;
            output[output.strides[0] * i +this->class_] += 1;
        };
    };

    virtual void mental_image(ArrayND<atomic<OUTPUT_VALUE_TYPE>>& output) {
        for(auto kv : this->memory) {
            for(int j = 0; j < this->tuple_lenght; ++j) {
                if ((kv.first >> j) & 1)
                    output(this->class_, this->mapping[j]) += kv.second;
                else
                    output(this->class_, this->mapping[j]) += -kv.second;
            };
        };
    };

    virtual uint64_t get_size() {
        return this->memory.size();
    };

    virtual void clear() {
        this->memory.clear();
    };

    // virtual void new_mapping(int* mapping) {
    //     delete [] this->mapping;
    //     this->mapping = mapping;
    // }

    virtual ~RamBase() {
        //delete [] this->mapping;
    };

};


template <class RAM_TYPE>
class DiscriminatorBase {
public:

    int input_lenght;
    int tuple_lenght;
    int num_rams;

    RAM_TYPE** rams;

    DiscriminatorBase(int input_lenght, int tuple_lenght, int class_, int** mapping) {
        this->input_lenght = input_lenght;
        this->tuple_lenght = tuple_lenght;
        int offset = input_lenght % tuple_lenght;
        this->num_rams = input_lenght / tuple_lenght + (offset > 0);
        this->rams = new RAM_TYPE*[this->num_rams];

        for(int i = 0; i < this->num_rams; ++i) {
            int ram_tuple_lenght = (i == this->num_rams - 1 && offset ? (offset) : (tuple_lenght));
            this->rams[i] = new RAM_TYPE(input_lenght, ram_tuple_lenght, mapping[i], class_);
        };
    };

    void clear() {
        for(int i = 0; i < this->num_rams; ++i)
            this->rams[i]->clear();
    };

    uint64_t get_size() {
        uint64_t size = 0;
        for(int i = 0; i < this->num_rams; ++i)
            size += this->rams[i]->get_size();
        return size;
    };

    virtual ~DiscriminatorBase() {
        for(int i = 0; i < this->num_rams; ++i) {
            delete this->rams[i];
        };
    };

};


template <class DISCRIMINATOR_TYPE, class RAM_TYPE, class OUTPUT_VALUE_TYPE>
class WiSARDBase {
public:

    int input_lenght;
    int tuple_lenght;
    int num_classes;
    int num_rams;
    bool canonical;
    bool pool_deleted;

    DISCRIMINATOR_TYPE** discriminators;

    int num_threads;
    thread_pool* pool;
    vector<vector<vector<int>>> thread_rams; // Thread, Discriminator, Ram
    
    WiSARDBase(int input_lenght, int tuple_lenght, int num_classes, bool canonical) {
        
        this->input_lenght = input_lenght;
        this->tuple_lenght = tuple_lenght;
        this->num_classes = num_classes;
        this->canonical = canonical;

        this->discriminators = new DISCRIMINATOR_TYPE*[num_classes];
        int** mapping = complete_mapping(input_lenght, tuple_lenght);
        for(int i = 0; i < num_classes; ++i) {
            if(!this->canonical) 
                mapping = complete_mapping(input_lenght, tuple_lenght);
            this->discriminators[i] = new DISCRIMINATOR_TYPE(input_lenght, tuple_lenght, i, mapping);
        };
        this->num_rams = this->discriminators[0]->num_rams;
        delete mapping;

        this->num_threads = std::thread::hardware_concurrency();        
        this->pool = new thread_pool(this->num_threads);
        this->pool->sleep_duration = 0;

        for(int i = 0; i < this->num_threads; ++i) {
            this->thread_rams.push_back(vector<vector<int>>());
            for(int  j = 0; j < this->num_classes; ++j) {
                this->thread_rams[i].push_back(vector<int>());
            };
        };
        int thread = 0;
        for(int i = 0; i < this->num_classes; ++i) {
            for(int j = 0; j < this->num_rams; ++j) {
                this->thread_rams[thread][i].push_back(j);
                thread = (thread + 1) % this->num_threads;     
            };   
        };

        this->pool_deleted = false;

    };

    void delete_ths() {
        delete this->pool;
        this->pool_deleted = true;
    };

    template <typename F, typename... A>
    void run_parallel(const F &task, const A &...args) {
        for(int i = 0; i < this->num_threads; ++i)
            this->pool->push_task(std::bind(task, i, this, args...));
        this->pool->wait_for_tasks();
    };   

    static void train_work(int thread, WiSARDBase* obj, ArrayND<bool>& input, ArrayND<int>& classes) {
        int ram;
        for(int i = 0; i < input.shape[0]; ++i) {
            for(size_t j = 0; j < obj->thread_rams[thread][classes[i]].size(); ++j) {
                ram = obj->thread_rams[thread][classes[i]][j];
                obj->discriminators[classes[i]]->rams[ram]->train(input, i);
            };
        };
    };

    virtual void train(ArrayND<bool>& input, ArrayND<int>& classes) {
        this->run_parallel(WiSARDBase::train_work, ref(input), ref(classes));
    };

    static void predict_work(int thread, WiSARDBase* obj, ArrayND<bool>& input, ArrayND<atomic<OUTPUT_VALUE_TYPE>>& output, int begin, int end) {
        int ram;
        for(int i = begin; i < end; ++i) {
            for(int c = 0; c < obj->num_classes; ++c) {
                for(size_t j = 0; j < obj->thread_rams[thread][c].size(); ++j) {
                    ram = obj->thread_rams[thread][c][j];
                    obj->discriminators[c]->rams[ram]->predict(input, output, i);
                };
            };
        };
    };

    virtual ArrayND<atomic<OUTPUT_VALUE_TYPE>> predict(ArrayND<bool>& input, int begin=0, int end=-1) {
        if (end == -1) end = input.shape[0];
        ArrayND<atomic<OUTPUT_VALUE_TYPE>> output({input.shape[0], this->num_classes});
        this->run_parallel(WiSARDBase::predict_work, ref(input), ref(output), begin, end);
        return output;
    };

    virtual void predict(ArrayND<bool>& input, ArrayND<atomic<OUTPUT_VALUE_TYPE>>& output, int begin=0, int end=-1) {
        if (end == -1) end = input.shape[0];
        this->run_parallel(WiSARDBase::predict_work, ref(input), ref(output), begin, end);
    };

    static void mental_images_work(int thread, WiSARDBase* obj, ArrayND<atomic<OUTPUT_VALUE_TYPE>>& output) {
        int ram;
        for(int c = 0; c < obj->num_classes; ++c) {
            for(size_t j = 0; j < obj->thread_rams[thread][c].size(); ++j) {
                ram = obj->thread_rams[thread][c][j];
                obj->discriminators[c]->rams[ram]->mental_image(output);
            };
        };
    };

    ArrayND<atomic<OUTPUT_VALUE_TYPE>> mental_images() {
        ArrayND<atomic<OUTPUT_VALUE_TYPE>> output({this->num_classes, this->input_lenght});
        this->run_parallel(WiSARDBase::mental_images_work, ref(output));
        return output;
    };

    uint64_t get_size() {
        uint64_t size = 0;
        for(int i = 0; i < this->num_classes; ++i)
            size += this->discriminators[i]->get_size();
        return size;
    };

    void clear() {
        for(int i = 0; i < this->num_classes; ++i)
            this->discriminators[i]->clear();
    };

    virtual ~WiSARDBase() {
        for(int i = 0; i < this->num_classes; ++i) {
            delete this->discriminators[i];
        };
        delete [] this->discriminators;
        if(!this->pool_deleted)
            delete this->pool;
    };

};


#endif