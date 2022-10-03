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

typedef struct classess {
    bool classes[2];
};

class Ram {
public:

    int input_lenght;
    int tuple_lenght;
    int* mapping;
    int class_;

    ska::flat_hash_map<uint64_t, classess> memory;

    Ram(int input_lenght, int tuple_lenght, int* mapping, int class_) {
        this->input_lenght = input_lenght;
        this->tuple_lenght = tuple_lenght;
        this->mapping = mapping;
        this->class_ = class_;
    }; 

    uint64_t get_addr(ArrayND<bool>& input, int i) {
        uint64_t addr = 0;
        for(int j = 0; j < this->tuple_lenght; ++j) { 
            addr |= input(i, this->mapping[j]) << j;
        };
        return addr;
    };

    virtual void train(ArrayND<bool>& input, int i, int class_) {
        this->memory[this->get_addr(input, i)].classes[class_] = 1;
    };

    virtual void predict(ArrayND<bool>& input, ArrayND<atomic<int>>& output, int i) {
        uint64_t addr = this->get_addr(input, i);
        if (this->memory.count(addr)) {
            classess c = this->memory[addr];
            for(int j = 0; j < 10; ++j) {
                // output(i, j) += c.classes[j];
                if (c.classes[j])  output(i, j)++;
            };
        };
    };

    virtual ~Ram() {
        
    };

};

class Discriminator {
public:

    int input_lenght;
    int tuple_lenght;
    int num_rams;

    Ram** rams;

    Discriminator(int input_lenght, int tuple_lenght, int class_, int** mapping) {
        this->input_lenght = input_lenght;
        this->tuple_lenght = tuple_lenght;
        int offset = input_lenght % tuple_lenght;
        this->num_rams = input_lenght / tuple_lenght + (offset > 0);
        this->rams = new Ram*[this->num_rams];

        for(int i = 0; i < this->num_rams; ++i) {
            int ram_tuple_lenght = (i == this->num_rams - 1 && offset ? (offset) : (tuple_lenght));
            this->rams[i] = new Ram(input_lenght, ram_tuple_lenght, mapping[i], class_);
        };
    };

    virtual ~Discriminator() {
        for(int i = 0; i < this->num_rams; ++i) {
            delete this->rams[i];
        };
    };

};

class WiSARD {
public:

    int input_lenght;
    int tuple_lenght;
    int num_classes;
    int num_rams;

    Discriminator** discriminators;

    int num_threads;
    thread_pool* pool;
    vector<vector<vector<int>>> thread_rams; // Thread, Discriminator, Ram
    
    WiSARD(int input_lenght, int tuple_lenght, int num_classes) {
        
        this->input_lenght = input_lenght;
        this->tuple_lenght = tuple_lenght;
        this->num_classes = num_classes;

        this->discriminators = new Discriminator*[1];
        int** mapping = complete_mapping(input_lenght, tuple_lenght);
        this->discriminators[0] = new Discriminator(input_lenght, tuple_lenght, 0, mapping);
        this->num_rams = this->discriminators[0]->num_rams;
        delete mapping;

    };

    ArrayND<bool> sequential_training_predict(ArrayND<bool>& input, ArrayND<int>& classes) {
        ArrayND<bool> output({input.shape[0], 2});
        for(int i = 1; i < classes.shape[0]; ++i) {
            for(int j = 0; j < this->num_rams; ++j)
                this->discriminators[0]->rams[j]->train(input, classes, i-1);
                this->discriminators[0]->rams[j]->predict(input, output, i);
        };
    };

    virtual ~WiSARD() {
        for(int i = 0; i < 1; ++i) {
            delete this->discriminators[i];
        };
        delete [] this->discriminators;
        delete this->pool;
    };

};