#include "mapping.h"
#include "utils.h"

#include <algorithm>
#include <random>
#include <chrono>

#include <iostream>

using namespace std;
using namespace chrono;

int** complete_mapping(int inp_size, int n_bits) {
    int* mapp = range(0, inp_size);
    int seed = system_clock::now().time_since_epoch().count();
    shuffle(mapp, mapp + inp_size, default_random_engine(seed)); 
    int** mapping = slice(mapp, inp_size, n_bits);
    delete [] mapp;
    return mapping;
};

int* random_mapping(int inp_size, int n_bits) {
    int* arange = range(0, inp_size);
    int* choice = random_choice(arange, inp_size, n_bits);
    delete []  arange;
    return choice;
};

int*** pc1D_complete_mapping(int x_dim, int z_dim, int n_bits) {
    int input_lenght = x_dim * z_dim;
    int offset = input_lenght % n_bits;
    int num_mappings = input_lenght / n_bits + (offset > 0);

    int** inputs = new int*[input_lenght];
    for(int x = 0; x < x_dim; ++x) {
        for(int z = 0; z < z_dim; ++z) {
            inputs[x*z_dim + z] = new int[2];
            inputs[x*z_dim + z][0] = x;
            inputs[x*z_dim + z][1] = z;
        };
    };
    
    int seed = system_clock::now().time_since_epoch().count();
    shuffle(inputs, inputs + input_lenght, default_random_engine(seed));
    
    int*** mapping = slice(inputs, input_lenght, n_bits);
    
    return mapping;
};

int*** pc2D_complete_mapping(int x_dim, int y_dim, int z_dim, int n_bits) {
    int input_lenght = x_dim * y_dim * z_dim;
    int offset = input_lenght % n_bits;
    int num_mappings = input_lenght / n_bits + (offset > 0);

    int** inputs = new int*[input_lenght];
    for(int x = 0; x < x_dim; ++x) {
        for(int y = 0; y < y_dim; ++y) {
            for(int z = 0; z < z_dim; ++z) {
                inputs[x*y_dim*z_dim + y*z_dim + z] = new int[3];
                inputs[x*y_dim*z_dim + y*z_dim + z][0] = x;
                inputs[x*y_dim*z_dim + y*z_dim + z][1] = y;
                inputs[x*y_dim*z_dim + y*z_dim + z][2] = z;
            };
        };
    };
    
    int seed = system_clock::now().time_since_epoch().count();
    shuffle(inputs, inputs + input_lenght, default_random_engine(seed));
    
    int*** mapping = slice(inputs, input_lenght, n_bits);

    return mapping;
};