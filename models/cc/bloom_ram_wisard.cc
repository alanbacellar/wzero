#include "base_classes.h"

#include "wisard.cc"

#include <iostream>
using namespace std;

class BloomRamFilter : public RamBase<uint64_t, bool, int> {
public:

    WiSARD* wisard;
    int ram_tuple_lenght;

    BloomRamFilter(int input_lenght, int tuple_lenght, int* mapping, int _class) : 
    RamBase(input_lenght, tuple_lenght, mapping, _class) {}; 

    void create_rams(int ram_tuple_lenght) {
        this->ram_tuple_lenght = ram_tuple_lenght;
        this->wisard = new WiSARD(this->tuple_lenght, ram_tuple_lenght, 1, true);
        this->wisard->delete_ths();
    };

    uint64_t get_addr(ArrayND<bool>& input, int i, int ram) {
       uint64_t addr = 0;
        for(int j = 0; j < this->wisard->discriminators[0]->rams[ram]->tuple_lenght; ++j)
            addr |= input(i, this->mapping[this->wisard->discriminators[0]->rams[ram]->mapping[j]]) << j;
        return addr; 
    };

    void train(ArrayND<bool>& input, int i) {
        uint64_t addr;
        for(int r = 0; r < this->wisard->num_rams; ++r) {
            addr = this->get_addr(input, i, r);
            this->wisard->discriminators[0]->rams[r]->memory[addr] = 1;
        };
    };

    void predict(ArrayND<bool>& input, ArrayND<atomic<int>>& output, int i) {
        uint64_t addr;
        for(int r = 0; r < this->wisard->num_rams; ++r) {
            addr = this->get_addr(input, i, r);
            if(!(this->wisard->discriminators[0]->rams[r]->memory.count(addr)))
                return;
        };
        output(i, this->class_)++;
    };

    virtual uint64_t get_size() {
        return 1 << this->ram_tuple_lenght;
    };

    virtual ~BloomRamFilter() {
        delete this->wisard;
    };
};


class BloomRamDiscriminator : public DiscriminatorBase<BloomRamFilter> {
public:

    BloomRamDiscriminator(int input_lenght, int tuple_lenght, int _class, int** mapping) :
    DiscriminatorBase(input_lenght, tuple_lenght, _class, mapping) {}

};


class BloomRamWiSARD : public WiSARDBase<BloomRamDiscriminator, BloomRamFilter, int> {
public:

    BloomRamWiSARD(int input_lenght, int tuple_lenght, int ram_tuple_lenght, int num_classes, bool canonical)
    : WiSARDBase(input_lenght, tuple_lenght, num_classes, canonical) {
        for(int i = 0; i < this->num_classes; ++i) {
            for(int j = 0; j < this->discriminators[i]->num_rams; ++j)
                this->discriminators[i]->rams[j]->create_rams(ram_tuple_lenght);
        };
    };
};