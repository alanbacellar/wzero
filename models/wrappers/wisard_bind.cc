#include "base_bind.h"

#include "../cc/wisard.cc" 

class BindWiSARD : public BindBase<WiSARD, int> {
public:

    BindWiSARD(int input_lenght, int tuple_lenght, int num_classes, bool canonical)
    : BindBase(input_lenght, tuple_lenght, num_classes, canonical) {};

};