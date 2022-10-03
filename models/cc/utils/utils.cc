#include <algorithm>
#include <random>
#include <chrono>

using namespace std;
using namespace chrono;

#include "utils.h"

#include <iostream>

int* range(int a, int b) {
    int* array  = new int[b-a];
    for (int i = 0; i < b-a; ++i) array[i] = a + i;
    return array;
};

int** slice(int* array, int lenght, int slice_lenght) {
    int slice_offset = lenght % slice_lenght;
    int num_slices = lenght / slice_lenght + (slice_offset > 0);
    int** new_array = new int*[num_slices];
    int i_arr = 0;
    int size;
    for(int i = 0; i < num_slices; ++i) {
        new_array[i] = new int[slice_lenght];
        size = ((i == num_slices - 1 && (slice_offset > 0)) ? (slice_offset) : (slice_lenght));
        for(int j = 0; j < size; ++j) {
            new_array[i][j] = array[i_arr]; 
            ++i_arr;
        };
    };
    return new_array;
};

int*** slice(int** array, int lenght, int slice_lenght) {
    int slice_offset = lenght % slice_lenght;
    int num_slices = lenght / slice_lenght + (slice_offset > 0);
    int*** new_array = new int**[num_slices];
    int i_arr = 0;
    int size;
    for(int i = 0; i < num_slices; ++i) {
        new_array[i] = new int*[slice_lenght];
        size = ((i == num_slices - 1 && (slice_offset > 0)) ? (slice_offset) : (slice_lenght));
        for(int j = 0; j < size; ++j) {
            new_array[i][j] = array[i_arr]; 
            ++i_arr;
        };
    };
    return new_array;
};

int* random_choice(int* array, int array_len, int len) {
    int* array_copy = new int[array_len];
    for(int i = 0; i < array_len; ++i) 
        array_copy[i] = array[i];

    int seed = system_clock::now().time_since_epoch().count();
    shuffle(array_copy, array_copy + array_len, default_random_engine(seed)); 

    int* choice = new int[len];
    for(int i = 0; i < len; ++i) 
        choice[i] = array_copy[i];
    
    delete [] array_copy;
    
    return choice;
};
