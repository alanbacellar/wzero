#include <random>
#include "base_classes.h"

class BloomFilter : public RamBase<uint64_t, bool, int> {
public:

    uint64_t** filter_keys;
    int num_filters;
    int filter_tuple_lenght;

    BloomFilter(int input_lenght, int tuple_lenght, int* mapping, int _class) : 
    RamBase(input_lenght, tuple_lenght, mapping, _class) {}; 

    void create_filter_keys(int num_filters, int filter_tuple_lenght) {

        this->num_filters = num_filters;
        this->filter_tuple_lenght = filter_tuple_lenght;

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<uint64_t> distrib(0, 1 << filter_tuple_lenght);

        this->filter_keys = new uint64_t*[num_filters];
        
        for(int i = 0; i < num_filters; ++i) {
            this->filter_keys[i] = new uint64_t[this->tuple_lenght];
            for(int j = 0; j < this->tuple_lenght; ++j) {
                this->filter_keys[i][j] = distrib(gen);
            };
        };

    };

    uint64_t get_addr(ArrayND<bool>& input, int i, int filter) {
        uint64_t addr = input(i, this->mapping[0]) * this->filter_keys[filter][0];
        for(int j = 1; j < this->tuple_lenght; ++j) {
            addr ^= input(i, this->mapping[j]) * this->filter_keys[filter][j];
        };
        return addr;
    };

    void train(ArrayND<bool>& input, int i) {
        for(int j = 0; j < this->num_filters; ++j)
            this->memory[this->get_addr(input, i, j)] = 1;
    };

    void predict(ArrayND<bool>& input, ArrayND<atomic<int>>& output, int i) {
        for(int j = 0; j < this->num_filters; ++j) {
            if(!(this->memory.count(this->get_addr(input, i, j))))
                return;
        };
        output(i, this->class_)++;
    };

    virtual uint64_t get_size() {
        return 1 << this->filter_tuple_lenght;
    };
};


class BloomDiscriminator : public DiscriminatorBase<BloomFilter> {
public:

    BloomDiscriminator(int input_lenght, int tuple_lenght, int _class, int** mapping) :
    DiscriminatorBase(input_lenght, tuple_lenght, _class, mapping) {}

};


class BloomWiSARD : public WiSARDBase<BloomDiscriminator, BloomFilter, int> {
public:

    BloomWiSARD(int input_lenght, int tuple_lenght, int num_filters, int filter_tuple_lenght, int num_classes, bool canonical)
    : WiSARDBase(input_lenght, tuple_lenght, num_classes, canonical) {
        for(int i = 0; i < this->num_classes; ++i) {
            for(int j = 0; j < this->discriminators[i]->num_rams; ++j)
                this->discriminators[i]->rams[j]->create_filter_keys(num_filters, filter_tuple_lenght);
        };
    };
};