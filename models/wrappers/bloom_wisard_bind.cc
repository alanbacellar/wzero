#include "base_bind.h"

#include "../cc/bloom_wisard.cc"

class BindBloomWiSARD : public BindBase<BloomWiSARD, int> {
public:

    BindBloomWiSARD(int input_lenght, int tuple_lenght, int num_filters, int filter_tuple_lenght, int num_classes, bool canonical)
    : BindBase(input_lenght, tuple_lenght, num_filters, filter_tuple_lenght, num_classes, canonical) {};

};