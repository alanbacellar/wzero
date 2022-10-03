#include "base_classes.h"

class ProbRam : public RamBase<uint64_t, int, int> {
public:

    int ram;

    ProbRam(int input_lenght, int tuple_lenght, int* mapping, int class_) : 
    RamBase(input_lenght, tuple_lenght, mapping, class_) {};

    void train(ArrayND<bool>& input, int i) {
        this->memory[this->get_addr(input, i)]++;
    };

    void predict(ArrayND<bool>& input, ArrayND<atomic<int>>& output, int i) {
        uint64_t addr = this->get_addr(input, i);
        if (this->memory.count(addr)) {
            // output(i, this->class_, this->ram) = this->memory[addr];
            output[output.strides[0]*i + output.strides[1]*this->class_ + this->ram] = this->memory[addr];
        };
    };

    void predict(ArrayND<bool>& input, ArrayND<atomic<int>>& output, int i, int bleaching) {
        uint64_t addr = this->get_addr(input, i);
        if (this->memory.count(addr)) {
            if(this->memory[addr] > bleaching)
                output[output.strides[0]*i + this->class_]++; 
        };
    };

};


class ProbDiscriminator : public DiscriminatorBase<ProbRam> {
public:

    ProbDiscriminator(int input_lenght, int tuple_lenght, int class_, int** mapping) :
    DiscriminatorBase(input_lenght, tuple_lenght, class_, mapping) {}

};


class ProbWiSARD : public WiSARDBase<ProbDiscriminator, ProbRam, int> {
public:

    ProbWiSARD(int input_lenght, int tuple_lenght, int num_classes, bool canonical)
    : WiSARDBase(input_lenght, tuple_lenght, num_classes, canonical) {
        for(int i = 0; i < this->num_classes; ++i) {
            for(int j = 0; j < this->num_rams; ++j)
                this->discriminators[i]->rams[j]->ram = j;
        };
    };

    ArrayND<float> predict2(ArrayND<bool>& input, int begin=0, int end=-1) {
        
        ArrayND<atomic<int>> ram_output({input.shape[0], this->num_classes, this->discriminators[0]->num_rams});
        ArrayND<float> output({input.shape[0], this->num_classes});


        WiSARDBase::predict(input, ram_output);

        this->pool->parallelize_loop(0, input.shape[0], [&input, &ram_output, &output, this](int a, int b) {

            float sum;

            for(int i = a; i < b; ++i) {
                
                for(int j = 0; j < this->num_classes; ++j)
                    output(i, j) = 1;

                for(int ram = 0; ram < this->num_rams; ++ram) {
                    sum = 0.0000001;
                    for(int j = 0; j < this->num_classes; ++j) {
                        sum += ram_output(i, j, ram) + 0.1;
                         if(ram % 20)
                            output(i, j) *= 10;
                    };
                    if (sum != 0.0000001) {
                        for(int j = 0; j < this->num_classes; ++j)
                            output(i, j) *=  (ram_output(i, j, ram) + 0.1) / sum;
                    };
                   
                };

                // for(int ram = 0; ram < this->num_rams; ++ram) {
                //     sum = 0.0000001;
                //     for(int j = 0; j < this->num_classes; ++j) {
                //         sum += ram_output(i, j, ram) + 0.1;
                //     };
                //     if (sum != 0.0000001) {
                //         for(int j = 0; j < this->num_classes; ++j)
                //             output(i, j) *=  1 - 0.000001*(1 - (ram_output(i, j, ram) + 0.1) / sum);
                //     };
                   
                // };
            
            }; 

        });       
        
        return output;

    };
    
};