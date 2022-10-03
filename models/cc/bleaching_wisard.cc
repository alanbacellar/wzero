#include "base_classes.h"

class BleachingRam : public RamBase<uint64_t, int, int> {
public:

    int* bleaching;

    int ram;

    BleachingRam(int input_lenght, int tuple_lenght, int* mapping, int class_) : 
    RamBase(input_lenght, tuple_lenght, mapping, class_) {};

    void train(ArrayND<bool>& input, int i) {
        this->memory[this->get_addr(input, i)]++;
    };

    // void predict(ArrayND<bool>& input, ArrayND<atomic<int>>& output, int i) {
    //     uint64_t addr = this->get_addr(input, i);
    //     if (this->memory.count(addr)) {
    //         if(this->memory[addr] > *(this->bleaching))
    //             output(i, this->class_)++; 
    //     };
    // };

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
                // output(i, this->class_)++; 
                output[output.strides[0]*i + this->class_]++;
        };
    };

};


class BleachingDiscriminator : public DiscriminatorBase<BleachingRam> {
public:

    BleachingDiscriminator(int input_lenght, int tuple_lenght, int class_, int** mapping) :
    DiscriminatorBase(input_lenght, tuple_lenght, class_, mapping) {}

};


class BleachingWiSARD : public WiSARDBase<BleachingDiscriminator, BleachingRam, int> {
public:

    int* bleaching;

    BleachingWiSARD(int input_lenght, int tuple_lenght, int num_classes, bool canonical)
    : WiSARDBase(input_lenght, tuple_lenght, num_classes, canonical) {

        this->bleaching = new int();
        
        for(int i = 0; i < this->num_classes; ++i) {
            for(int j = 0; j < this->discriminators[i]->num_rams; ++j)
                this->discriminators[i]->rams[j]->bleaching = this->bleaching;
        };

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

    // ArrayND<atomic<int>> predict(ArrayND<bool>& input, int begin=0, int end=-1) {
        
    //     ArrayND<atomic<int>> output({input.shape[0], this->num_classes});
    //     WiSARDBase::predict(input, output);

    //     this->pool->parallelize_loop(0, input.shape[0],
    //                                 [&input, &output, this](int a, int b) {
            
    //         bool draw;
    //         int max;
    //         int bleaching = 0;

    //         for(int i = a; i < b; ++i) {

    //             draw = true;

    //             while(draw) {

    //                 draw = false;
    //                 max = output(i, 0);
                    
    //                 for(int j = 1; j < this->num_classes; ++j) {
    //                     if (output(i, j) == max)
    //                         draw = true;
    //                     else if (output(i, j) > max) {
    //                         draw = false;
    //                         max = output(i, j);
    //                     };
    //                 };

    //                 if (max == 0)
    //                     break;

    //                 if (draw) {
    //                     bleaching++;
    //                     for(int j = 0; j < this->num_classes; ++j) {
    //                         output(i, j) = 0;
    //                     };
    //                     this->predict_single(input, output, i, bleaching);
    //                 };
                    
    //             };

    //             bleaching = 0;
            
    //         }; 

    //     });       

        
    
    //     return output;

    // };

    ArrayND<atomic<int>> predict(ArrayND<bool>& input, int begin=0, int end=-1) {
        
        ArrayND<atomic<int>> ram_output({input.shape[0], this->num_classes, this->discriminators[0]->num_rams});
        ArrayND<atomic<int>> output({input.shape[0], this->num_classes});


        WiSARDBase::predict(input, ram_output);

        this->pool->parallelize_loop(0, input.shape[0],
                                    [&input, &ram_output, &output, this](int a, int b) {
            
            bool draw;
            int max;
            int bleaching = 0;

            for(int i = a; i < b; ++i) {

                draw = true;

                for(int j = 0; j < this->num_classes; ++j) {
                    for(int ram = 0; ram < this->num_rams; ++ram)
                        output(i, j) += ram_output(i, j, ram) > bleaching;
                };

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
                            for(int ram = 0; ram < this->num_rams; ++ram)
                                output(i, j) += ram_output(i, j, ram) > bleaching;
                        };
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