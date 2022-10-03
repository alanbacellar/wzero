#include "base_bind.h"

#include "../cc/bloom_ram_wisard.cc"

class BindBloomRamWiSARD : public BindBase<BloomRamWiSARD, int> {
public:

    BindBloomRamWiSARD(int input_lenght, int tuple_lenght, int ram_tuple_lenght, int num_classes, bool canonical)
    : BindBase(input_lenght, tuple_lenght, ram_tuple_lenght, num_classes, canonical) {};

};