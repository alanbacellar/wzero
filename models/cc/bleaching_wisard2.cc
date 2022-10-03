#include "base_classes.h"

class BleachingRam2 : public RamBase<uint64_t, int, int> {
public:

    int ram;

    BleachingRam2(int input_lenght, int tuple_lenght, int* mapping, int class_) : 
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


class BleachingDiscriminator2 : public DiscriminatorBase<BleachingRam2> {
public:

    BleachingDiscriminator2(int input_lenght, int tuple_lenght, int class_, int** mapping) :
    DiscriminatorBase(input_lenght, tuple_lenght, class_, mapping) {}

};


class BleachingWiSARD2 : public WiSARDBase<BleachingDiscriminator2, BleachingRam2, int> {
public:

    BleachingWiSARD2(int input_lenght, int tuple_lenght, int num_classes, bool canonical)
    : WiSARDBase(input_lenght, tuple_lenght, num_classes, canonical) {
        for(int i = 0; i < this->num_classes; ++i) {
            for(int j = 0; j < this->num_rams; ++j)
                this->discriminators[i]->rams[j]->ram = j;
        };
    };

    void predict_single(ArrayND<bool>& input, ArrayND<atomic<int>>& output, int i, int bleaching) {
        for(int d = 0; d < this->num_classes; ++d) {
            for(int r = 0; r < this->num_rams; ++r)
                this->discriminators[d]->rams[r]->predict(input, output, i, bleaching);
        };
    };

    ArrayND<int> predict2(ArrayND<bool>& input, int begin=0, int end=-1) {
        
        ArrayND<atomic<int>> ram_output({input.shape[0], this->num_classes, this->discriminators[0]->num_rams});
        ArrayND<int> output({input.shape[0], this->num_classes});


        WiSARDBase::predict(input, ram_output);

        // ArrayND<int> ram_output({input.shape[0], this->num_classes, this->discriminators[0]->num_rams});
        // for(int i = 0; i < ram_output2.size; ++i) {
        //     if(ram_output2[i] != 0)
        //     ram_output[i] = ram_output2[i];
        // };

        this->pool->parallelize_loop(0, input.shape[0],
                                    [&input, &ram_output, &output, this](int a, int b) {
            
            int max;
            int max_2;
            int max_class;

            for(int i = a; i < b; ++i) {

                max = 0;
                max_2 = 0;
                max_class = 0;

                for(int j = 0; j < this->num_classes; ++j)
                    std::sort((int*)ram_output.data + i*ram_output.strides[0] + j*ram_output.strides[1], (int*)ram_output.data + i*ram_output.strides[0] + j*ram_output.strides[1] + this->num_rams);

                for(int ram = 0; ram < this->num_rams; ++ram) {

                    for(int j = 0; j < this->num_classes; ++j) {
                        if (ram_output(i, j, ram) >= max) {
                            max_2 = max;
                            max = ram_output(i, j, ram);
                            max_class = j;
                        }
                        else if (output(i, j, ram) > max_2) {
                            max_2 = ram_output(i, j, ram);
                        };
                    };

                    if (ram == this->num_rams - 1)
                        break;
                    
                    if (max != max_2)// && max != ram_output(i, max_class, ram+1))
                        break;

                };

                for(int j = 0; j < this->num_classes; ++j) {
                    for(int ram = 0; ram < this->num_rams; ++ram)
                        output(i, j) += ram_output(i, j, ram) > max_2;
                };
            
            }; 

        });       
        
        return output;

    };
    
};