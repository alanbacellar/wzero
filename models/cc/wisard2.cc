#include <unordered_map>
#include <vector>
#include <atomic>

#include "utils/utils.h"
#include "utils/mapping.h"
#include "utils/array.h"
#include "utils/atomic_operators.h"

#include "include/thread_pool.hpp"
#include "include/flat_hash_map.hpp"

using namespace std;

typedef struct classess {
    bool classes[10];
};

class Ram2 {
public:

    int input_lenght;
    int tuple_lenght;
    int* mapping;
    int class_;

    int counter;

    ska::flat_hash_map<uint64_t, classess> memory;

    Ram2(int input_lenght, int tuple_lenght, int* mapping, int class_) {
        this->input_lenght = input_lenght;
        this->tuple_lenght = tuple_lenght;
        this->mapping = mapping;
        this->class_ = class_;

        this->counter = 0;
    }; 

    uint64_t get_addr(ArrayND<bool>& input, int i) {
        uint64_t addr = 0;
        for(int j = 0; j < this->tuple_lenght; ++j) { 
            addr |= input(i, this->mapping[j]) << j;
            // addr |= input[input.strides[0] * i + this->mapping[j]] << j;
        };

        // int j = 15;
        // if(this->counter < 2) {
        //     this->counter++;
        //     for(int k = 0; k < 15; ++k)
        //         cout << input(i, this->mapping[k]) << " ";
        //     cout << endl;
        //     cout << i << " " << this->mapping[j] << " "  << input(i, this->mapping[j]) << " " << j << endl;
        // };
        
        return addr;
    };

    void train(ArrayND<bool>& input, int i, int class_) {
        this->memory[this->get_addr(input, i)].classes[class_] = 1;
    };

    void predict(ArrayND<bool>& input, ArrayND<atomic<int>>& output, int i) {
        uint64_t addr = this->get_addr(input, i);
        if (this->memory.count(addr)) {
            classess c = this->memory[addr];
            for(int j = 0; j < 10; ++j) {
                // output(i, j) += c.classes[j];
                if (c.classes[j])  output(i, j)++;
                // if (c.classes[j])  output[output.strides[0] * i + j]++;
            };
        };
    };

    ~Ram2() {
        
    };

};

class Discriminator2 {
public:

    int input_lenght;
    int tuple_lenght;
    int num_rams;

    Ram2** rams;

    Discriminator2(int input_lenght, int tuple_lenght, int class_, int** mapping) {
        this->input_lenght = input_lenght;
        this->tuple_lenght = tuple_lenght;
        int offset = input_lenght % tuple_lenght;
        this->num_rams = input_lenght / tuple_lenght + (offset > 0);
        this->rams = new Ram2*[this->num_rams];

        for(int i = 0; i < this->num_rams; ++i) {
            int ram_tuple_lenght = (i == this->num_rams - 1 && offset ? (offset) : (tuple_lenght));
            this->rams[i] = new Ram2(input_lenght, ram_tuple_lenght, mapping[i], class_);
        };
    };

    ~Discriminator2() {
        for(int i = 0; i < this->num_rams; ++i) {
            delete this->rams[i];
        };
    };

};

class WiSARD2 {
public:

    int input_lenght;
    int tuple_lenght;
    int num_classes;
    int num_rams;

    Discriminator2** discriminators;

    int num_threads;
    thread_pool* pool;
    vector<vector<vector<int>>> thread_rams; // Thread, Discriminator, Ram
    
    WiSARD2(int input_lenght, int tuple_lenght, int num_classes) {
        
        this->input_lenght = input_lenght;
        this->tuple_lenght = tuple_lenght;
        this->num_classes = num_classes;

        this->discriminators = new Discriminator2*[1];
        int** mapping = complete_mapping(input_lenght, tuple_lenght);
        //for(int i = 0; i < num_classes; ++i) {
            //if(!this->canonical) 
            //    mapping = complete_mapping(input_lenght, tuple_lenght);
            this->discriminators[0] = new Discriminator2(input_lenght, tuple_lenght, 0, mapping);
        //};
        this->num_rams = this->discriminators[0]->num_rams;
        delete mapping;

        this->num_threads = std::thread::hardware_concurrency();        
        this->pool = new thread_pool(this->num_threads);
        this->pool->sleep_duration = 0;

        for(int i = 0; i < this->num_threads; ++i) {
            this->thread_rams.push_back(vector<vector<int>>());
            for(int  j = 0; j < 1; ++j) {
                this->thread_rams[i].push_back(vector<int>());
            };
        };
        int thread = 0;
        for(int i = 0; i < 1; ++i) {
            for(int j = 0; j < this->num_rams; ++j) {
                this->thread_rams[thread][i].push_back(j);
                thread = (thread + 1) % this->num_threads;     
            };   
        };

    };

    template <typename F, typename... A>
    void run_parallel(const F &task, const A &...args) {
        for(int i = 0; i < this->num_threads; ++i)
            this->pool->push_task(std::bind(task, i, this, args...));
        this->pool->wait_for_tasks();
    };   

    static void train_work(int thread, WiSARD2* obj, ArrayND<bool>& input, ArrayND<int>& classes) {
        int ram;
        for(int i = 0; i < input.shape[0]; ++i) {
            for(size_t j = 0; j < obj->thread_rams[thread][0].size(); ++j) {
                ram = obj->thread_rams[thread][0][j];
                obj->discriminators[0]->rams[ram]->train(input, i, classes[i]);
            };
        };
    };

    void train(ArrayND<bool>& input, ArrayND<int>& classes) {
        this->run_parallel(WiSARD2::train_work, ref(input), ref(classes));
    };

    static void predict_work(int thread, WiSARD2* obj, ArrayND<bool>& input, ArrayND<atomic<int>>& output, int begin, int end) {
        int ram;
        for(int i = begin; i < end; ++i) {
            for(int c = 0; c < 1; ++c) {
                for(size_t j = 0; j < obj->thread_rams[thread][c].size(); ++j) {
                    ram = obj->thread_rams[thread][c][j];
                    obj->discriminators[c]->rams[ram]->predict(input, output, i);
                };
            };
        };
    };

    ArrayND<atomic<int>> predict(ArrayND<bool>& input, int begin=0, int end=-1) {
        if (end == -1) end = input.shape[0];
        ArrayND<atomic<int>> output({input.shape[0], this->num_classes});
        this->run_parallel(WiSARD2::predict_work, ref(input), ref(output), begin, end);
        return output;
    };

    void predict(ArrayND<bool>& input, ArrayND<atomic<int>>& output, int begin=0, int end=-1) {
        if (end == -1) end = input.shape[0];
        this->run_parallel(WiSARD2::predict_work, ref(input), ref(output), begin, end);
    };

    ~WiSARD2() {
        for(int i = 0; i < 1; ++i) {
            delete this->discriminators[i];
        };
        delete [] this->discriminators;
        delete this->pool;
    };

};


// #include <unordered_map>
// #include <vector>
// #include <atomic>

// #include "utils/utils.h"
// #include "utils/mapping.h"
// #include "utils/array.h"
// #include "utils/atomic_operators.h"

// #include "include/thread_pool.hpp"
// #include "include/flat_hash_map.hpp"

// using namespace std;

// typedef struct classess {
//     bool classes[10];
// };

// class Ram2 {
// public:

//     int input_lenght;
//     int tuple_lenght;
//     int* mapping;
//     int class_;

//     ska::flat_hash_map<uint64_t, classess> memory;

//     Ram2(int input_lenght, int tuple_lenght, int* mapping, int class_) {
//         this->input_lenght = input_lenght;
//         this->tuple_lenght = tuple_lenght;
//         this->mapping = mapping;
//         this->class_ = class_;
//     }; 

//     uint64_t* get_addrs(ArrayND<bool>& input) {
//         uint64_t* addrs = new uint64_t[input.shape[0]]();
//         for(int i = 0; i < input.shape[0]; ++i) {
//             for(int j = 0; j < this->tuple_lenght; ++j) { 
//                 addrs[i] |= input(i, this->mapping[j]) << j;
//             };
//         };
//         return addrs;
//     };

//     virtual void train(ArrayND<bool>& input, ArrayND<int> classes) {
//         uint64_t* addrs = this->get_addrs(input);
//         for(int i = 0; i < input.shape[0]; ++i)
//             this->memory[addrs[i]].classes[classes[i]] = 1;
//     };

//     virtual void predict(ArrayND<bool>& input, ArrayND<atomic<int>>& output) {
//         uint64_t* addrs = this->get_addrs(input);
//         for(int i = 0; i < input.shape[0]; ++i) {
//             if (this->memory.count(addrs[i])) {
//                 classess c = this->memory[addrs[i]];
//                 for(int j = 0; j < 10; ++j) {
//                     // output(i, j) += c.classes[j];
//                     if (c.classes[j])  output(i, j)++;
//                 };
//             };
//         };
//     };

//     virtual ~Ram2() {
        
//     };

// };

// class Discriminator2 {
// public:

//     int input_lenght;
//     int tuple_lenght;
//     int num_rams;

//     Ram2** rams;

//     Discriminator2(int input_lenght, int tuple_lenght, int class_, int** mapping) {
//         this->input_lenght = input_lenght;
//         this->tuple_lenght = tuple_lenght;
//         int offset = input_lenght % tuple_lenght;
//         this->num_rams = input_lenght / tuple_lenght + (offset > 0);
//         this->rams = new Ram2*[this->num_rams];

//         for(int i = 0; i < this->num_rams; ++i) {
//             int ram_tuple_lenght = (i == this->num_rams - 1 && offset ? (offset) : (tuple_lenght));
//             this->rams[i] = new Ram2(input_lenght, ram_tuple_lenght, mapping[i], class_);
//         };
//     };

//     virtual ~Discriminator2() {
//         for(int i = 0; i < this->num_rams; ++i) {
//             delete this->rams[i];
//         };
//     };

// };

// class WiSARD2 {
// public:

//     int input_lenght;
//     int tuple_lenght;
//     int num_classes;
//     int num_rams;

//     Discriminator2** discriminators;

//     int num_threads;
//     thread_pool* pool;
//     vector<vector<vector<int>>> thread_rams; // Thread, Discriminator, Ram
    
//     WiSARD2(int input_lenght, int tuple_lenght, int num_classes) {
        
//         this->input_lenght = input_lenght;
//         this->tuple_lenght = tuple_lenght;
//         this->num_classes = num_classes;

//         this->discriminators = new Discriminator2*[1];
//         int** mapping = complete_mapping(input_lenght, tuple_lenght);
//         //for(int i = 0; i < num_classes; ++i) {
//             //if(!this->canonical) 
//             //    mapping = complete_mapping(input_lenght, tuple_lenght);
//             this->discriminators[0] = new Discriminator2(input_lenght, tuple_lenght, 0, mapping);
//         //};
//         this->num_rams = this->discriminators[0]->num_rams;
//         delete mapping;

//         this->num_threads = std::thread::hardware_concurrency();        
//         this->pool = new thread_pool(this->num_threads);

//         for(int i = 0; i < this->num_threads; ++i) {
//             this->thread_rams.push_back(vector<vector<int>>());
//             for(int  j = 0; j < 1; ++j) {
//                 this->thread_rams[i].push_back(vector<int>());
//             };
//         };
//         int thread = 0;
//         for(int i = 0; i < 1; ++i) {
//             for(int j = 0; j < this->num_rams; ++j) {
//                 this->thread_rams[thread][i].push_back(j);
//                 thread = (thread + 1) % this->num_threads;     
//             };   
//         };

//     };

//     template <typename F, typename... A>
//     void run_parallel(const F &task, const A &...args) {
//         for(int i = 0; i < this->num_threads; ++i)
//             this->pool->push_task(std::bind(task, i, this, args...));
//         this->pool->wait_for_tasks();
//     };   

//     static void train_work(int thread, WiSARD2* obj, ArrayND<bool>& input, ArrayND<int>& classes) {
//         int ram;
//         for(size_t j = 0; j < obj->thread_rams[thread][0].size(); ++j) {
//             ram = obj->thread_rams[thread][0][j];
//             obj->discriminators[0]->rams[ram]->train(input, classes);
//         };
//     };

//     virtual void train(ArrayND<bool>& input, ArrayND<int>& classes) {
//         this->run_parallel(WiSARD2::train_work, ref(input), ref(classes));
//     };

//     static void predict_work(int thread, WiSARD2* obj, ArrayND<bool>& input, ArrayND<atomic<int>>& output, int begin, int end) {
//         int ram;
//         for(int c = 0; c < 1; ++c) {
//             for(size_t j = 0; j < obj->thread_rams[thread][c].size(); ++j) {
//                 ram = obj->thread_rams[thread][c][j];
//                 obj->discriminators[c]->rams[ram]->predict(input, output);
//             };
//         };
//     };

//     virtual ArrayND<atomic<int>> predict(ArrayND<bool>& input, int begin=0, int end=-1) {
//         if (end == -1) end = input.shape[0];
//         ArrayND<atomic<int>> output(input.shape[0], this->num_classes);
//         this->run_parallel(WiSARD2::predict_work, ref(input), ref(output), begin, end);
//         return output;
//     };

//     virtual void predict(ArrayND<bool>& input, ArrayND<atomic<int>>& output, int begin=0, int end=-1) {
//         if (end == -1) end = input.shape[0];
//         this->run_parallel(WiSARD2::predict_work, ref(input), ref(output), begin, end);
//     };

//     virtual ~WiSARD2() {
//         for(int i = 0; i < 1; ++i) {
//             delete this->discriminators[i];
//         };
//         delete [] this->discriminators;
//         delete this->pool;
//     };

// };
