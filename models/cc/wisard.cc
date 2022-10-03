#include "base_classes.h"

#ifndef WISARD_CC
#define WISARD_CC

class Ram : public RamBase<uint64_t, bool, int> {
public:

    Ram(int input_lenght, int tuple_lenght, int* mapping, int class_) : 
    RamBase(input_lenght, tuple_lenght, mapping, class_) {}; 

};


class Discriminator : public DiscriminatorBase<Ram> {
public:

    Discriminator(int input_lenght, int tuple_lenght, int class_, int** mapping) :
    DiscriminatorBase(input_lenght, tuple_lenght, class_, mapping) {}

};


class WiSARD : public WiSARDBase<Discriminator, Ram, int> {
public:

    WiSARD(int input_lenght, int tuple_lenght, int num_classes, bool canonical)
    : WiSARDBase(input_lenght, tuple_lenght, num_classes, canonical) {};
};

#endif