#ifndef UTILS_H
#define UTILS_H

int* range(int a, int b);

int** slice(int* array, int lenght, int slice_lenght);

int* random_choice(int* array, int array_len, int len);

int*** slice(int** array, int lenght, int slice_lenght);

#endif