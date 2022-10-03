#ifndef ARRAY_H
#define ARRAY_H

#include <vector>

template <class T>
class ArrayND {
public:

    T* data;
    int size;
    int num_dims;
    int* shape;
    int* strides;
    bool allocated_memory;

    void init(T* data, std::vector<int> dims) {
        
        this->num_dims = dims.size();
        this->shape = new int[this->num_dims];
        this->strides = new int[this->num_dims];
        this->size = 1;

        for(int i = this->num_dims-1; i >= 0; --i) {
            this->strides[i] = this->size;
            this->shape[i] = dims[i];
            this->size *= dims[i];
        };

        if (data == nullptr) {
            this->data = new T[this->size]();
            this->allocated_memory = true;
        } else {
            this->data = data;
            this->allocated_memory = false;
        };

    };

    ArrayND(std::vector<int> dims) {
        this->init(nullptr, dims);
    };

    ArrayND(T* data, std::vector<int> dims) {
        this->init(data, dims);
    };

    ArrayND() = default;

    T& operator[](int i) {
        return this->data[i];
    };

    T& operator()(int a) {
        return this->data[a];
    };

    T& operator()(int a, int b) {
        return this->data[a*this->strides[0] + b*this->strides[1]];
    };

    T& operator()(int a, int b, int c) { 
        return this->data[a*this->strides[0] + b*this->strides[1] + c*this->strides[2]];
    };

    T& operator()(int a, int b, int c, int d) {
        return this->data[a*this->strides[0] + b*this->strides[1] + c*this->strides[2] + d*this->strides[3]];
    };

    T& operator()(int a, int b, int c, int d, int e) {
        return this->data[a*this->strides[0] + b*this->strides[1] + c*this->strides[2] + d*this->strides[3] + e*this->strides[4]];
    };

    T& operator()(int a, int b, int c, int d, int e, int f) {
        return this->data[a*this->strides[0] + b*this->strides[1] + c*this->strides[2] + d*this->strides[3] + e*this->strides[4] + f*this->strides[5]];
    };

    template <typename ...A>
    T& operator()(int a, int b, int c, int d, int e, int f, A... args) {
        
        int indexs[this->num_dims] = {a, b, c, d, e, f, args...};
        int index = 0; 								
		
        for(int i = 0; i < this->num_dims; ++i)
            index += this->strides[i] * indexs[i];

        return this->data[index];
    };

    // template <typename ...A>
    // T& operator()(A... args) {
        
    //     int indexs[this->num_dims] = {args...};
    //     int index = 0; 								
		
    //     for(int i = 0; i < this->num_dims; ++i)
    //         index += this->strides[i] * indexs[i];

    //     return this->data[index];
    // };
    

    T sum() {
        T sum = 0;
        for(int i = 0; i < this->size; ++i)
            sum += this->data[i];
        return sum;
    };

    ~ArrayND() {
        delete this->shape;
        if (this->allocated_memory)
            delete [] this->data;
    };

};


#endif