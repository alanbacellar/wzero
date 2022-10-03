#include <random>
#include "base_classes.h"

class BleachingBloomFilter : public RamBase<uint64_t, int, int> {
public:

    uint64_t** filter_keys;
    int num_filters;
    int filter_tuple_lenght;
    int* bleaching;

    BleachingBloomFilter(int input_lenght, int tuple_lenght, int* mapping, int _class) : 
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
            if (input(i, this->mapping[j]))
                addr ^= this->filter_keys[filter][j];
        };
        return addr;
    };

    void train(ArrayND<bool>& input, int i) {
        for(int j = 0; j < this->num_filters; ++j)
            this->memory[this->get_addr(input, i, j)]++;
    };

    void predict(ArrayND<bool>& input, ArrayND<atomic<int>>& output, int i) {
        uint64_t addr;
        for(int j = 0; j < this->num_filters; ++j) {
            addr = this->get_addr(input, i, j);
            if(!(this->memory.count(addr)))
                return;
            if(!(this->memory[addr] > *(this->bleaching)))
                return;
        };
        output(i, this->class_)++;
    };

    void predict(ArrayND<bool>& input, ArrayND<atomic<int>>& output, int i, int bleaching) {
        uint64_t addr;
        for(int j = 0; j < this->num_filters; ++j) {
            addr = this->get_addr(input, i, j);
            if(!(this->memory.count(addr)))
                return;
            if(!(this->memory[addr] > bleaching))
                return;
        };
        output(i, this->class_)++;
    };

    virtual uint64_t get_size() {
        return 1 << this->filter_tuple_lenght;
    };


};


class BleachingBloomDiscriminator : public DiscriminatorBase<BleachingBloomFilter> {
public:

    BleachingBloomDiscriminator(int input_lenght, int tuple_lenght, int _class, int** mapping) :
    DiscriminatorBase(input_lenght, tuple_lenght, _class, mapping) {}

};


class BleachingBloomWiSARD : public WiSARDBase<BleachingBloomDiscriminator, BleachingBloomFilter, int> {
public:

    int* bleaching;

    BleachingBloomWiSARD(int input_lenght, int tuple_lenght, int num_filters, int filter_tuple_lenght, int num_classes, bool canonical)
    : WiSARDBase(input_lenght, tuple_lenght, num_classes, canonical) {
        
        this->bleaching = new int();
        
        for(int i = 0; i < this->num_classes; ++i) {
            for(int j = 0; j < this->discriminators[i]->num_rams; ++j)
                this->discriminators[i]->rams[j]->bleaching = this->bleaching;
        };

        for(int i = 0; i < this->num_classes; ++i) {
            for(int j = 0; j < this->discriminators[i]->num_rams; ++j)
                this->discriminators[i]->rams[j]->create_filter_keys(num_filters, filter_tuple_lenght);
        };
    };

    void predict_single(ArrayND<bool>& input, ArrayND<atomic<int>>& output, int i, int bleaching) {
        for(int d = 0; d < this->num_classes; ++d) {
            for(int r = 0; r < this->num_rams; ++r)
                this->discriminators[d]->rams[r]->predict(input, output, i, bleaching);
        };
    };

    ArrayND<atomic<int>> predict(ArrayND<bool>& input, int begin=0, int end=-1) {
        
        ArrayND<atomic<int>> output({input.shape[0], this->num_classes});
        WiSARDBase::predict(input, output);

        this->pool->parallelize_loop(0, input.shape[0],
                                    [&input, &output, this](int a, int b) {
            
            bool draw;
            int max;
            int bleaching = 0;

            for(int i = a; i < b; ++i) {

                draw = true;

                while(draw) {

                    draw = false;
                    max = output(i, 0);
                    
                    for(int j = 1; j < this->num_classes; ++j) {
                        if (output(i, j) == max)
                            draw = true;
                        else if (output(i, j) > max) {
                            draw = false;
                            max = output(i, j);
                        };
                    };

                    if (max == 0)
                        break;

                    if (draw) {
                        bleaching++;
                        for(int j = 0; j < this->num_classes; ++j) {
                            output(i, j) = 0;
                        };
                        this->predict_single(input, output, i, bleaching);
                    };
                    
                };

                bleaching = 0;
            
            }; 

        });       

        
    
        return output;

    };

    ArrayND<atomic<int>> predictb(ArrayND<bool>& input, int bleaching) {
        *(this->bleaching) = bleaching; 
        ArrayND<atomic<int>> output = WiSARDBase::predict(input);
        *(this->bleaching) = 0; 
        return output;
    };
};