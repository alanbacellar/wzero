#include <unordered_map>
#include <vector>
#include <atomic>

#include "utils/utils.h"
#include "utils/mapping.h"
#include "utils/array.h"
#include "utils/atomic_operators.h"

#include "include/thread_pool.hpp"
#include "include/flat_hash_map.hpp"

#include <iostream>

using namespace std;

typedef struct classessPC2D {
    bool classes[10];
};

class RamPC2D {
public:

    int input_lenght;
    int tuple_lenght;
    int** mapping;
    int class_;

    int x_dim;
    int y_dim;
    int window_size;
    int stride;

    ska::flat_hash_map<uint64_t, classessPC2D> memory;

    RamPC2D(int input_lenght, int tuple_lenght, int** mapping, int class_) {
        this->input_lenght = input_lenght;
        this->tuple_lenght = tuple_lenght;
        this->mapping = mapping;
        this->class_ = class_;
    }; 

    void init(int x_dim, int y_dim, int window_size, int stride) {
        this->x_dim = x_dim;
        this->y_dim = y_dim;
        this->window_size = window_size;
        this->stride = stride;
    };

    uint64_t get_addr(ArrayND<bool>& input, int i, int x, int y) {
        uint64_t addr = 0;
        for(int j = 0; j < this->tuple_lenght; ++j) { 
            addr |= input(i, this->mapping[j][0] + x, this->mapping[j][1] + y, this->mapping[j][2]) << j;
        };
        return addr;
    };

    virtual void train(ArrayND<bool>& input, int i, int class_) {
        for(int x = 0; x < this->x_dim - this->window_size + 1; x = x + this->stride) {
            for(int y = 0; y < this->y_dim - this->window_size + 1; y = y + this->stride) {
                this->memory[this->get_addr(input, i, x, y)].classes[class_] = 1;
            };
        }; 
    };

    virtual void predict(ArrayND<bool>& input, ArrayND<atomic<int>>& output, int i) {
        uint64_t addr;
        for(int x = 0; x < this->x_dim - this->window_size + 1; x = x + this->stride) {
            for(int y = 0; y < this->y_dim - this->window_size + 1; y = y + this->stride) {
                addr = this->get_addr(input, i, x, y);
                if (this->memory.count(addr)) {
                    classessPC2D c = this->memory[addr];
                    for(int j = 0; j < 10; ++j) {
                        // output(i, j) += c.classes[j];
                        if (c.classes[j]) output(i, j)++;
                    };
                };
            };
        };
    };

    virtual ~RamPC2D() {
        
    };

};

class DiscriminatorPC2D {
public:

    int input_lenght;
    int tuple_lenght;
    int num_rams;

    RamPC2D** rams;

    DiscriminatorPC2D(int input_lenght, int window_lenght, int tuple_lenght, int class_, int*** mapping) {
        this->input_lenght = input_lenght;
        this->tuple_lenght = tuple_lenght;
        int offset = input_lenght % tuple_lenght;
        this->num_rams = input_lenght / tuple_lenght + (offset > 0);
        this->rams = new RamPC2D*[this->num_rams];

        for(int i = 0; i < this->num_rams; ++i) {
            int ram_tuple_lenght = (i == this->num_rams - 1 && offset ? (offset) : (tuple_lenght));
            this->rams[i] = new RamPC2D(window_lenght, ram_tuple_lenght, mapping[i], class_);
        };
    };

    virtual ~DiscriminatorPC2D() {
        for(int i = 0; i < this->num_rams; ++i) {
            delete this->rams[i];
        };
    };

};

class WiSARDPC2D {
public:

    int input_lenght;
    int tuple_lenght;
    int num_classes;
    int num_rams;

    DiscriminatorPC2D** discriminators;

    int num_threads;
    thread_pool* pool;
    vector<vector<vector<int>>> thread_rams; // Thread, Discriminator, Ram

    
    WiSARDPC2D(int x_dim, int y_dim, int z_dim, int window_size, int stride, int tuple_lenght, int num_classes) {
        
        this->input_lenght = NULL;
        this->tuple_lenght = tuple_lenght;
        this->num_classes = num_classes;

        this->discriminators = new DiscriminatorPC2D*[1];
        int*** mapping = pc2D_complete_mapping(window_size, window_size, z_dim, tuple_lenght);
        this->discriminators[0] = new DiscriminatorPC2D(window_size*window_size*z_dim, window_size*window_size, tuple_lenght, 0, mapping);
        this->num_rams = this->discriminators[0]->num_rams;
        for(int i = 0; i < this->num_rams; ++i)
            this->discriminators[0]->rams[i]->init(x_dim, y_dim, window_size, stride);
        //delete mapping;
        this->num_threads = std::thread::hardware_concurrency();        
        this->pool = new thread_pool(this->num_threads);

        for(int i = 0; i < this->num_threads; ++i) {
            this->thread_rams.push_back(vector<vector<int>>());
            for(int  j = 0; j < 1; ++j) {
                this->thread_rams[i].push_back(vector<int>());
            };
        };

        int thread = 0;
        for(int i = 0; i < 1; ++i) {
            for(int j = 0; j < this->num_rams; ++j) {
                this->thread_rams[thread][i].push_back(j);
                thread = (thread + 1) % this->num_threads;     
            };   
        };

    };

    template <typename F, typename... A>
    void run_parallel(const F &task, const A &...args) {
        for(int i = 0; i < this->num_threads; ++i)
            this->pool->push_task(std::bind(task, i, this, args...));
        this->pool->wait_for_tasks();
    };   

    static void train_work(int thread, WiSARDPC2D* obj, ArrayND<bool>& input, ArrayND<int>& classes) {
        int ram;
        for(int i = 0; i < input.shape[0]; ++i) {
            for(size_t j = 0; j < obj->thread_rams[thread][0].size(); ++j) {
                ram = obj->thread_rams[thread][0][j];
                obj->discriminators[0]->rams[ram]->train(input, i, classes[i]);
            };
        };
    };

    virtual void train(ArrayND<bool>& input, ArrayND<int>& classes) {
        this->run_parallel(WiSARDPC2D::train_work, ref(input), ref(classes));
    };

    static void predict_work(int thread, WiSARDPC2D* obj, ArrayND<bool>& input, ArrayND<atomic<int>>& output, int begin, int end) {
        int ram;
        for(int i = begin; i < end; ++i) {
            for(int c = 0; c < 1; ++c) {
                for(size_t j = 0; j < obj->thread_rams[thread][c].size(); ++j) {
                    ram = obj->thread_rams[thread][c][j];
                    obj->discriminators[c]->rams[ram]->predict(input, output, i);
                };
            };
        };
    };

    virtual ArrayND<atomic<int>> predict(ArrayND<bool>& input, int begin=0, int end=-1) {
        if (end == -1) end = input.shape[0];
        ArrayND<atomic<int>> output({input.shape[0], this->num_classes});
        this->run_parallel(WiSARDPC2D::predict_work, ref(input), ref(output), begin, end);
        return output;
    };

    virtual void predict(ArrayND<bool>& input, ArrayND<atomic<int>>& output, int begin=0, int end=-1) {
        if (end == -1) end = input.shape[0];
        this->run_parallel(WiSARDPC2D::predict_work, ref(input), ref(output), begin, end);
    };

    virtual ~WiSARDPC2D() {
        for(int i = 0; i < 1; ++i) {
            delete this->discriminators[i];
        };
        delete [] this->discriminators;
        delete this->pool;
    };

};