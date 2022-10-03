#ifndef MAPPING_H
#define MAPPING_H

int** complete_mapping(int inp_size, int n_bits);

int* random_mapping(int inp_size, int n_bits);

int*** pc1D_complete_mapping(int x_dim, int z_dim, int n_bits);

int*** pc2D_complete_mapping(int x_dim, int y_dim, int z_dim, int n_bits);

#endif