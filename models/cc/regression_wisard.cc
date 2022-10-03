#include <unordered_map>
#include "base_classes.h"

class RegressionRam : public RamBase<uint64_t, float, float> {
public:

    unordered_map<uint64_t, int> counters;

    RegressionRam(int input_lenght, int tuple_lenght, int* mapping, int class_) : 
    RamBase(input_lenght, tuple_lenght, mapping, class_) {};

    
    using RamBase::train;

    void train(ArrayND<bool>& input, ArrayND<float>& classes, int i) {
        uint64_t addr = this->get_addr(input, i);
        this->memory[addr] +=  classes(i, this->class_);
        this->counters[addr]++;
    }

    using RamBase::predict;

    void predict(ArrayND<bool>& input, ArrayND<atomic<float>>& output, ArrayND<atomic<int>>& output_counter, int i) {
        uint64_t addr = this->get_addr(input, i);
        if(this->memory.count(addr)) {
            output(i, this->class_) += this->memory[addr] / this->counters[addr];
            output_counter(i, this->class_) += 1;    
        };
    };
    
    void clear() {
        this->memory.clear();
        this->counters.clear();
    };

    ~RegressionRam() {};

};


class RegressionDiscriminator : public DiscriminatorBase<RegressionRam> {
public:

    RegressionDiscriminator(int input_lenght, int tuple_lenght, int class_, int** mapping) :
    DiscriminatorBase(input_lenght, tuple_lenght, class_, mapping) {};

    ~RegressionDiscriminator() {};

};


class RegressionWiSARD : public WiSARDBase<RegressionDiscriminator, RegressionRam, float> {
public:

    RegressionWiSARD(int input_lenght, int tuple_lenght, int num_classes, bool canonical)
    : WiSARDBase(input_lenght, tuple_lenght, num_classes, canonical) {};

    static void train_work(int thread, WiSARDBase* obj, ArrayND<bool>& input,ArrayND<float>& classes) {
        int ram;
        for(int i = 0; i < input.shape[0]; ++i) {
            for(int c = 0; c < obj->num_classes; ++c) {
                for(size_t j = 0; j < obj->thread_rams[thread][c].size(); ++j) {
                    ram = obj->thread_rams[thread][c][j];
                    obj->discriminators[c]->rams[ram]->train(input, classes, i);
                };
            };
        };
    };

    void train(ArrayND<bool>& input, ArrayND<float>& classes) {
        this->run_parallel(RegressionWiSARD::train_work, ref(input), ref(classes));
    };

    static void predict_work(int thread, WiSARDBase* obj, ArrayND<bool>& input, ArrayND<atomic<float>>& output, ArrayND<atomic<int>>& output_counter, int begin, int end) {
        int ram;
        for(int i = begin; i < end; ++i) {
            for(int c = 0; c < obj->num_classes; ++c) {
                for(size_t j = 0; j < obj->thread_rams[thread][c].size(); ++j) {
                    ram = obj->thread_rams[thread][c][j];
                    obj->discriminators[c]->rams[ram]->predict(input, output, output_counter, i);
                };
            };
        };
    };

    ArrayND<atomic<float>> predict(ArrayND<bool>& input, int begin=0, int end=-1) {
        if (end == -1) end = input.shape[0];
        ArrayND<atomic<float>> output({input.shape[0], this->num_classes});
        ArrayND<atomic<int>> output_counter({input.shape[0], this->num_classes});
        this->run_parallel(RegressionWiSARD::predict_work, ref(input), ref(output), ref(output_counter), begin, end);
        this->pool->parallelize_loop(begin * output.shape[1], end * output.shape[1],
                                    [&output, &output_counter](int a, int b) {
                                        for(int i = a; i < b; ++i) {
                                            if (output_counter[i] != 0)
                                                output[i] = output[i] / output_counter[i];
                                        };
                                    }                          
        );
        return output;
    };

    ~RegressionWiSARD() {};

};