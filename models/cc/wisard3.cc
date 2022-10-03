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

// class Ram3 {
// public:

//     int input_lenght;
//     int tuple_lenght;
//     int* mapping;
//     int class_;

//     ska::flat_hash_map<uint64_t, bool*> memory;

//     Ram3(int input_lenght, int tuple_lenght, int* mapping, int class_) {
//         this->input_lenght = input_lenght;
//         this->tuple_lenght = tuple_lenght;
//         this->mapping = mapping;
//         this->class_ = class_;
//     }; 

//     uint64_t get_addr(ArrayND<bool>& input, int i) {
//         uint64_t addr = 0;
//         for(int j = 0; j < this->tuple_lenght; ++j) { 
//             addr |= input(i, this->mapping[j]) << j;
//         };
//         return addr;
//     };

//     virtual void train(ArrayND<bool>& input, int i, int class_) {
//         uint64_t addr = this->get_addr(input, i);
//         if(!this->memory.count(addr))
//             this->memory[addr] = (bool*)calloc(10, sizeof(bool));
//         this->memory[addr][class_] = 1;
//     };

//     virtual void predict(ArrayND<bool>& input, ArrayND<atomic<int>>& output, int i) {
//         uint64_t addr = this->get_addr(input, i);
//         if (this->memory.count(addr)) {
//             bool* c = this->memory[addr];
//             for(int j = 0; j < 10; ++j) {
//                 // output(i, j) += c[j];
//                 if (c[j])  output(i, j)++;
//             };
//         };
//     };

//     virtual ~Ram3() {
        
//     };

// };


// class Discriminator3 {
// public:

//     int input_lenght;
//     int tuple_lenght;
//     int num_rams;

//     Ram3** rams;

//     Discriminator3(int input_lenght, int tuple_lenght, int class_, int** mapping) {
//         this->input_lenght = input_lenght;
//         this->tuple_lenght = tuple_lenght;
//         int offset = input_lenght % tuple_lenght;
//         this->num_rams = input_lenght / tuple_lenght + (offset > 0);
//         this->rams = new Ram3*[this->num_rams];

//         for(int i = 0; i < this->num_rams; ++i) {
//             int ram_tuple_lenght = (i == this->num_rams - 1 && offset ? (offset) : (tuple_lenght));
//             this->rams[i] = new Ram3(input_lenght, ram_tuple_lenght, mapping[i], class_);
//         };
//     };

//     virtual ~Discriminator3() {
//         for(int i = 0; i < this->num_rams; ++i) {
//             delete this->rams[i];
//         };
//     };

// };

// class WiSARD3 {
// public:

//     int input_lenght;
//     int tuple_lenght;
//     int num_classes;
//     int num_rams;

//     Discriminator3** discriminators;

//     int num_threads;
//     thread_pool* pool;
//     vector<vector<vector<int>>> thread_rams; // Thread, Discriminator, Ram
    
//     WiSARD3(int input_lenght, int tuple_lenght, int num_classes) {
        
//         this->input_lenght = input_lenght;
//         this->tuple_lenght = tuple_lenght;
//         this->num_classes = num_classes;

//         this->discriminators = new Discriminator3*[1];
//         int** mapping = complete_mapping(input_lenght, tuple_lenght);
//         //for(int i = 0; i < num_classes; ++i) {
//             //if(!this->canonical) 
//             //    mapping = complete_mapping(input_lenght, tuple_lenght);
//             this->discriminators[0] = new Discriminator3(input_lenght, tuple_lenght, 0, mapping);
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

//     static void train_work(int thread, WiSARD3* obj, ArrayND<bool>& input, ArrayND<int>& classes) {
//         int ram;
//         for(int i = 0; i < input.shape[0]; ++i) {
//             for(size_t j = 0; j < obj->thread_rams[thread][0].size(); ++j) {
//                 ram = obj->thread_rams[thread][0][j];
//                 obj->discriminators[0]->rams[ram]->train(input, i, classes[i]);
//             };
//         };
//     };

//     virtual void train(ArrayND<bool>& input, ArrayND<int>& classes) {
//         this->run_parallel(WiSARD3::train_work, ref(input), ref(classes));
//     };

//     static void predict_work(int thread, WiSARD3* obj, ArrayND<bool>& input, ArrayND<atomic<int>>& output, int begin, int end) {
//         int ram;
//         for(int i = begin; i < end; ++i) {
//             for(int c = 0; c < 1; ++c) {
//                 for(size_t j = 0; j < obj->thread_rams[thread][c].size(); ++j) {
//                     ram = obj->thread_rams[thread][c][j];
//                     obj->discriminators[c]->rams[ram]->predict(input, output, i);
//                 };
//             };
//         };
//     };

//     virtual ArrayND<atomic<int>> predict(ArrayND<bool>& input, int begin=0, int end=-1) {
//         if (end == -1) end = input.shape[0];
//         ArrayND<atomic<int>> output(input.shape[0], this->num_classes);
//         this->run_parallel(WiSARD3::predict_work, ref(input), ref(output), begin, end);
//         return output;
//     };

//     virtual void predict(ArrayND<bool>& input, ArrayND<atomic<int>>& output, int begin=0, int end=-1) {
//         if (end == -1) end = input.shape[0];
//         this->run_parallel(WiSARD3::predict_work, ref(input), ref(output), begin, end);
//     };

//     virtual ~WiSARD3() {
//         for(int i = 0; i < 1; ++i) {
//             delete this->discriminators[i];
//         };
//         delete [] this->discriminators;
//         delete this->pool;
//     };

// };


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

// class Node {
// public:
//     Node* right;
//     Node* left;
//     uint64_t key;
//     int diff_bit;
//     bool classes[10];

//     Node() {
//         this->right = NULL;
//         this->left = NULL;
//         diff_bit = -1;
//     };

//     ~Node() {

//     };

// };

// class Ram3 {
// public:

//     int input_lenght;
//     int tuple_lenght;
//     int* mapping;
//     int class_;
//     Node* root;
//     bool root_null;

//     Ram3(int input_lenght, int tuple_lenght, int* mapping, int class_) {
//         this->input_lenght = input_lenght;
//         this->tuple_lenght = tuple_lenght;
//         this->mapping = mapping;
//         this->class_ = class_;
//         this->root = new Node();
//         this->root_null = true;
//     }; 

//     uint64_t get_addr(ArrayND<bool>& input, int i) {
//         uint64_t addr = 0;
//         for(int j = 0; j < this->tuple_lenght; ++j) { 
//             addr |= input(i, this->mapping[j]) << j;
//         };
//         return addr;
//     };

//     virtual void train(ArrayND<bool>& input, int i, int class_) {
//         uint64_t addr = 0;
//         bool input_bit;
//         bool leaf_bit;
//         Node* leaf = this->root;
//         int j;

//         if(this->root_null) {
//             addr = this->get_addr(input, i);
//             leaf->key = addr;
//             leaf->classes[class_] = 1;
//             this->root_null = false;
//             return;
//         };
        
//         for(j = this->tuple_lenght - 1; j >= 0; --j) {
//             input_bit = input(i, this->mapping[j]);
//             addr |= input_bit << j;
//             leaf_bit = (leaf->key >> j) & 1;
//             if(j == leaf->diff_bit || input_bit != leaf_bit) {
//                 if(input_bit) {
//                     if (leaf->right == NULL)
//                         break;    
//                     leaf = leaf->right;
//                 } 
//                 else {
//                     if (leaf->left == NULL)
//                         break;
//                     leaf = leaf->left;
//                 };
//             };
//         };

//         for(int k = j-1; k >= 0; --k) {
//             addr |= input(i, this->mapping[k]) << k;
//         };

//         if(addr != leaf->key) {
//             if(j > leaf->diff_bit)
//                 leaf->diff_bit = j;
//             if(input_bit) {
//                 leaf->right = new Node();
//                 leaf = leaf->right;
//             }
//             else {
//                 leaf->left = new Node();
//                 leaf = leaf->left;
//             };
//             leaf->key = addr;
//         };

//         leaf->classes[class_] = 1;
    
//     };

//     virtual void predict(ArrayND<bool>& input, ArrayND<atomic<int>>& output, int i) {
//         uint64_t addr = 0;
//         bool input_bit;
//         bool leaf_bit;
//         Node* leaf = this->root;
//         int j;

//         if(this->root_null)
//             return;
        
//         for(j = this->tuple_lenght - 1; j >= 0; --j) {
//             input_bit = input(i, this->mapping[j]);
//             addr |= input_bit << j;
//             leaf_bit = (leaf->key >> j) & 1;
//             if(j == leaf->diff_bit || input_bit != leaf_bit) {
//                 if(input_bit) {
//                     if (leaf->right == NULL)
//                         return;    
//                     leaf = leaf->right;
//                 } 
//                 else {
//                     if (leaf->left == NULL)
//                         return;
//                     leaf = leaf->left;
//                 };
//             };
//         };

//         for(j = 0; j < 10; ++j) {
//             if (leaf->classes[j])  output(i, j)++;
//         };

//     };

//     virtual ~Ram3() {
        
//     };

// };

// class Discriminator3 {
// public:

//     int input_lenght;
//     int tuple_lenght;
//     int num_rams;

//     Ram3** rams;

//     Discriminator3(int input_lenght, int tuple_lenght, int class_, int** mapping) {
//         this->input_lenght = input_lenght;
//         this->tuple_lenght = tuple_lenght;
//         int offset = input_lenght % tuple_lenght;
//         this->num_rams = input_lenght / tuple_lenght + (offset > 0);
//         this->rams = new Ram3*[this->num_rams];

//         for(int i = 0; i < this->num_rams; ++i) {
//             int ram_tuple_lenght = (i == this->num_rams - 1 && offset ? (offset) : (tuple_lenght));
//             this->rams[i] = new Ram3(input_lenght, ram_tuple_lenght, mapping[i], class_);
//         };
//     };

//     virtual ~Discriminator3() {
//         for(int i = 0; i < this->num_rams; ++i) {
//             delete this->rams[i];
//         };
//     };

// };

// class WiSARD3 {
// public:

//     int input_lenght;
//     int tuple_lenght;
//     int num_classes;
//     int num_rams;

//     Discriminator3** discriminators;

//     int num_threads;
//     thread_pool* pool;
//     vector<vector<vector<int>>> thread_rams; // Thread, Discriminator, Ram
    
//     WiSARD3(int input_lenght, int tuple_lenght, int num_classes) {
        
//         this->input_lenght = input_lenght;
//         this->tuple_lenght = tuple_lenght;
//         this->num_classes = num_classes;

//         this->discriminators = new Discriminator3*[1];
//         int** mapping = complete_mapping(input_lenght, tuple_lenght);
//         //for(int i = 0; i < num_classes; ++i) {
//             //if(!this->canonical) 
//             //    mapping = complete_mapping(input_lenght, tuple_lenght);
//             this->discriminators[0] = new Discriminator3(input_lenght, tuple_lenght, 0, mapping);
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

//     static void train_work(int thread, WiSARD3* obj, ArrayND<bool>& input, ArrayND<int>& classes) {
//         int ram;
//         for(int i = 0; i < input.shape[0]; ++i) {
//             for(size_t j = 0; j < obj->thread_rams[thread][0].size(); ++j) {
//                 ram = obj->thread_rams[thread][0][j];
//                 obj->discriminators[0]->rams[ram]->train(input, i, classes[i]);
//             };
//         };
//     };

//     virtual void train(ArrayND<bool>& input, ArrayND<int>& classes) {
//         this->run_parallel(WiSARD3::train_work, ref(input), ref(classes));
//     };

//     static void predict_work(int thread, WiSARD3* obj, ArrayND<bool>& input, ArrayND<atomic<int>>& output, int begin, int end) {
//         int ram;
//         for(int i = begin; i < end; ++i) {
//             for(int c = 0; c < 1; ++c) {
//                 for(size_t j = 0; j < obj->thread_rams[thread][c].size(); ++j) {
//                     ram = obj->thread_rams[thread][c][j];
//                     obj->discriminators[c]->rams[ram]->predict(input, output, i);
//                 };
//             };
//         };
//     };

//     virtual ArrayND<atomic<int>> predict(ArrayND<bool>& input, int begin=0, int end=-1) {
//         if (end == -1) end = input.shape[0];
//         ArrayND<atomic<int>> output(input.shape[0], this->num_classes);
//         this->run_parallel(WiSARD3::predict_work, ref(input), ref(output), begin, end);
//         return output;
//     };

//     virtual void predict(ArrayND<bool>& input, ArrayND<atomic<int>>& output, int begin=0, int end=-1) {
//         if (end == -1) end = input.shape[0];
//         this->run_parallel(WiSARD3::predict_work, ref(input), ref(output), begin, end);
//     };

//     virtual ~WiSARD3() {
//         for(int i = 0; i < 1; ++i) {
//             delete this->discriminators[i];
//         };
//         delete [] this->discriminators;
//         delete this->pool;
//     };

// };


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

// typedef struct classess3 {
//     bool classes[10];
// };

// class Ram3 {
// public:

//     int input_lenght;
//     int tuple_lenght;
//     int* mapping;
//     int class_;

//     ska::flat_hash_map<uint64_t, classess3> memory;

//     Ram3(int input_lenght, int tuple_lenght, int* mapping, int class_) {
//         this->input_lenght = input_lenght;
//         this->tuple_lenght = tuple_lenght;
//         this->mapping = mapping;
//         this->class_ = class_;
//     }; 

//     uint64_t get_addr(ArrayND<bool>& input, int i) {
//         uint64_t addr = 0;
//         for(int j = 0; j < this->tuple_lenght; ++j) { 
//             addr |= input(i, this->mapping[j]) << j;
//         };
//         return addr;
//     };

//     virtual void train(ArrayND<bool>& input, int i, int class_) {
//         this->memory[this->get_addr(input, i)].classes[class_] = 1;
//     };

//     virtual void predict(ArrayND<bool>& input, ArrayND<atomic<int>>& output, int i) {
//         uint64_t addr = this->get_addr(input, i);
//         if (this->memory.count(addr)) {
//             classess3 c = this->memory[addr];
//             for(int j = 0; j < 10; ++j) {
//                 // output(i, j) += c.classes[j];
//                 if (c.classes[j])  output(i, j)++;
//             };
//         };
//     };

//     virtual ~Ram3() {
        
//     };

// };

// class Discriminator3 {
// public:

//     int input_lenght;
//     int tuple_lenght;
//     int num_rams;

//     Ram3** rams;

//     Discriminator3(int input_lenght, int tuple_lenght, int class_, int** mapping) {
//         this->input_lenght = input_lenght;
//         this->tuple_lenght = tuple_lenght;
//         int offset = input_lenght % tuple_lenght;
//         this->num_rams = input_lenght / tuple_lenght + (offset > 0);
//         this->rams = new Ram3*[this->num_rams];

//         for(int i = 0; i < this->num_rams; ++i) {
//             int ram_tuple_lenght = (i == this->num_rams - 1 && offset ? (offset) : (tuple_lenght));
//             this->rams[i] = new Ram3(input_lenght, ram_tuple_lenght, mapping[i], class_);
//         };
//     };

//     virtual ~Discriminator3() {
//         for(int i = 0; i < this->num_rams; ++i) {
//             delete this->rams[i];
//         };
//     };

// };

// class WiSARD3 {
// public:

//     int input_lenght;
//     int tuple_lenght;
//     int num_classes;
//     int num_rams;

//     Discriminator3** discriminators;

//     int num_threads;
//     thread_pool* pool;
    
//     WiSARD3(int input_lenght, int tuple_lenght, int num_classes) {
        
//         this->input_lenght = input_lenght;
//         this->tuple_lenght = tuple_lenght;
//         this->num_classes = num_classes;

//         this->discriminators = new Discriminator3*[1];
//         int** mapping = complete_mapping(input_lenght, tuple_lenght);
//         //for(int i = 0; i < num_classes; ++i) {
//             //if(!this->canonical) 
//             //    mapping = complete_mapping(input_lenght, tuple_lenght);
//             this->discriminators[0] = new Discriminator3(input_lenght, tuple_lenght, 0, mapping);
//         //};
//         this->num_rams = this->discriminators[0]->num_rams;
//         delete mapping;

//         this->num_threads = std::thread::hardware_concurrency();        
//         this->pool = new thread_pool(this->num_threads);

//     };

//     template <typename F, typename... A>
//     void run_parallel(const F &task, const A &...args) {
//         for(int i = 0; i < this->discriminators[0]->num_rams; ++i)
//             this->pool->push_task(std::bind(task, i, this, args...));
//         this->pool->wait_for_tasks();
//     };   

//     static void train_work(int ram, WiSARD3* obj, ArrayND<bool>& input, ArrayND<int>& classes) {
//         for(int i = 0; i < input.shape[0]; ++i)
//             obj->discriminators[0]->rams[ram]->train(input, i, classes[i]);
//     };

//     virtual void train(ArrayND<bool>& input, ArrayND<int>& classes) {
//         this->run_parallel(WiSARD3::train_work, ref(input), ref(classes));
//     };

//     static void predict_work(int ram, WiSARD3* obj, ArrayND<bool>& input, ArrayND<atomic<int>>& output, int begin, int end) {
//         for(int i = begin; i < end; ++i)
//             obj->discriminators[0]->rams[ram]->predict(input, output, i);
//     };

//     virtual ArrayND<atomic<int>> predict(ArrayND<bool>& input, int begin=0, int end=-1) {
//         if (end == -1) end = input.shape[0];
//         ArrayND<atomic<int>> output(input.shape[0], this->num_classes);
//         this->run_parallel(WiSARD3::predict_work, ref(input), ref(output), begin, end);
//         return output;
//     };

//     virtual void predict(ArrayND<bool>& input, ArrayND<atomic<int>>& output, int begin=0, int end=-1) {
//         if (end == -1) end = input.shape[0];
//         this->run_parallel(WiSARD3::predict_work, ref(input), ref(output), begin, end);
//     };

//     virtual ~WiSARD3() {
//         for(int i = 0; i < 1; ++i) {
//             delete this->discriminators[i];
//         };
//         delete [] this->discriminators;
//         delete this->pool;
//     };

// };


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

// typedef struct classess3 {
//     bool classes[10];
// };

// class Ram3 {
// public:

//     int input_lenght;
//     int tuple_lenght;
//     int* mapping;
//     int class_;

//     ska::flat_hash_map<uint64_t, classess3> memory;

//     Ram3(int input_lenght, int tuple_lenght, int* mapping, int class_) {
//         this->input_lenght = input_lenght;
//         this->tuple_lenght = tuple_lenght;
//         this->mapping = mapping;
//         this->class_ = class_;
//     }; 

//     uint64_t get_addr(ArrayND<bool>& input, int i) {
//         uint64_t addr = 0;
//         for(int j = 0; j < this->tuple_lenght; ++j) { 
//             addr |= input(i, this->mapping[j]) << j;
//         };
//         return addr;
//     };

//     virtual void train(ArrayND<bool>& input, int i, int class_) {
//         this->memory[this->get_addr(input, i)].classes[class_] = 1;
//     };

//     virtual void predict(ArrayND<bool>& input, ArrayND<atomic<int>>& output, int i) {
//         uint64_t addr = this->get_addr(input, i);
//         if (this->memory.count(addr)) {
//             classess3 c = this->memory[addr];
//             for(int j = 0; j < 10; ++j) {
//                 // output(i, j) += c.classes[j];
//                 if (c.classes[j])  output(i, j)++;
//             };
//         };
//     };

//     virtual ~Ram3() {
        
//     };

// };

// class Discriminator3 {
// public:

//     int input_lenght;
//     int tuple_lenght;
//     int num_rams;

//     Ram3** rams;

//     Discriminator3(int input_lenght, int tuple_lenght, int class_, int** mapping) {
//         this->input_lenght = input_lenght;
//         this->tuple_lenght = tuple_lenght;
//         int offset = input_lenght % tuple_lenght;
//         this->num_rams = input_lenght / tuple_lenght + (offset > 0);
//         this->rams = new Ram3*[this->num_rams];

//         for(int i = 0; i < this->num_rams; ++i) {
//             int ram_tuple_lenght = (i == this->num_rams - 1 && offset ? (offset) : (tuple_lenght));
//             this->rams[i] = new Ram3(input_lenght, ram_tuple_lenght, mapping[i], class_);
//         };
//     };

//     virtual ~Discriminator3() {
//         for(int i = 0; i < this->num_rams; ++i) {
//             delete this->rams[i];
//         };
//     };

// };

// class WiSARD3 {
// public:

//     int input_lenght;
//     int tuple_lenght;
//     int num_classes;
//     int num_rams;

//     Discriminator3** discriminators;

//     int num_threads;
//     thread_pool* pool;
//     vector<vector<vector<int>>> thread_rams; // Thread, Discriminator, Ram
    
//     WiSARD3(int input_lenght, int tuple_lenght, int num_classes) {
        
//         this->input_lenght = input_lenght;
//         this->tuple_lenght = tuple_lenght;
//         this->num_classes = num_classes;

//         this->discriminators = new Discriminator3*[1];
//         int** mapping = complete_mapping(input_lenght, tuple_lenght);
//         //for(int i = 0; i < num_classes; ++i) {
//             //if(!this->canonical) 
//             //    mapping = complete_mapping(input_lenght, tuple_lenght);
//             this->discriminators[0] = new Discriminator3(input_lenght, tuple_lenght, 0, mapping);
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

//     static void train_work(int thread, WiSARD3* obj, ArrayND<bool>& input, ArrayND<int>& classes) {
//         int ram;
//         for(size_t j = 0; j < obj->thread_rams[thread][0].size(); ++j) {
//             ram = obj->thread_rams[thread][0][j];
//             for(int i = 0; i < input.shape[0]; ++i) {
//                 obj->discriminators[0]->rams[ram]->train(input, i, classes[i]);
//             };
//         };
//     };

//     virtual void train(ArrayND<bool>& input, ArrayND<int>& classes) {
//         this->run_parallel(WiSARD3::train_work, ref(input), ref(classes));
//     };

//     static void predict_work(int thread, WiSARD3* obj, ArrayND<bool>& input, ArrayND<atomic<int>>& output, int begin, int end) {
//         int ram;
//         for(int c = 0; c < 1; ++c) {
//             for(size_t j = 0; j < obj->thread_rams[thread][c].size(); ++j) {
//                 ram = obj->thread_rams[thread][c][j];
//                 for(int i = begin; i < end; ++i) {
//                     obj->discriminators[c]->rams[ram]->predict(input, output, i);
//                 };
//             };
//         };
//     };

//     virtual ArrayND<atomic<int>> predict(ArrayND<bool>& input, int begin=0, int end=-1) {
//         if (end == -1) end = input.shape[0];
//         ArrayND<atomic<int>> output(input.shape[0], this->num_classes);
//         this->run_parallel(WiSARD3::predict_work, ref(input), ref(output), begin, end);
//         return output;
//     };

//     virtual void predict(ArrayND<bool>& input, ArrayND<atomic<int>>& output, int begin=0, int end=-1) {
//         if (end == -1) end = input.shape[0];
//         this->run_parallel(WiSARD3::predict_work, ref(input), ref(output), begin, end);
//     };

//     virtual ~WiSARD3() {
//         for(int i = 0; i < 1; ++i) {
//             delete this->discriminators[i];
//         };
//         delete [] this->discriminators;
//         delete this->pool;
//     };

// };


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

typedef struct classess3 {
    bool classes[10];
};

class Ram3 {
public:

    int input_lenght;
    int tuple_lenght;
    int* mapping;
    int class_;

    ska::flat_hash_map<uint64_t, classess3> memory;

    Ram3(int input_lenght, int tuple_lenght, int* mapping, int class_) {
        this->input_lenght = input_lenght;
        this->tuple_lenght = tuple_lenght;
        this->mapping = mapping;
        this->class_ = class_;
    }; 

    uint64_t get_addr(ArrayND<bool>& input, int i) {
        uint64_t addr = 0;
        for(int j = 0; j < this->tuple_lenght; ++j) { 
            addr |= input(i, this->mapping[j]) << j;
        };
        return addr;
    };

    virtual void train(ArrayND<bool>& input, int i, int class_) {
        this->memory[this->get_addr(input, i)].classes[class_] = 1;
    };

    virtual void predict(ArrayND<bool>& input, ArrayND<atomic<int>>& output, int i) {
        uint64_t addr = this->get_addr(input, i);
        if (this->memory.count(addr)) {
            classess3 c = this->memory[addr];
            for(int j = 0; j < 10; ++j) {
                // output(i, j) += c.classes[j];
                if (c.classes[j])  output(i, j)++;
            };
        };
    };

    virtual ~Ram3() {
        
    };

};

class Discriminator3 {
public:

    int input_lenght;
    int tuple_lenght;
    int num_rams;

    Ram3** rams;

    Discriminator3(int input_lenght, int tuple_lenght, int class_, int** mapping) {
        this->input_lenght = input_lenght;
        this->tuple_lenght = tuple_lenght;
        int offset = input_lenght % tuple_lenght;
        this->num_rams = input_lenght / tuple_lenght + (offset > 0);
        this->rams = new Ram3*[this->num_rams];

        for(int i = 0; i < this->num_rams; ++i) {
            int ram_tuple_lenght = (i == this->num_rams - 1 && offset ? (offset) : (tuple_lenght));
            this->rams[i] = new Ram3(input_lenght, ram_tuple_lenght, mapping[i], class_);
        };
    };

    virtual ~Discriminator3() {
        for(int i = 0; i < this->num_rams; ++i) {
            delete this->rams[i];
        };
    };

};

class WiSARD3 {
public:

    int input_lenght;
    int tuple_lenght;
    int num_classes;
    int num_rams;

    Discriminator3** discriminators;

    int num_threads;
    thread_pool* pool;
    
    WiSARD3(int input_lenght, int tuple_lenght, int num_classes) {
        
        this->input_lenght = input_lenght;
        this->tuple_lenght = tuple_lenght;
        this->num_classes = num_classes;

        this->discriminators = new Discriminator3*[1];
        int** mapping = complete_mapping(input_lenght, tuple_lenght);
        //for(int i = 0; i < num_classes; ++i) {
            //if(!this->canonical) 
            //    mapping = complete_mapping(input_lenght, tuple_lenght);
            this->discriminators[0] = new Discriminator3(input_lenght, tuple_lenght, 0, mapping);
        //};
        this->num_rams = this->discriminators[0]->num_rams;
        delete mapping;

        this->num_threads = std::thread::hardware_concurrency();        
        this->pool = new thread_pool(this->num_threads);

    };

    template <typename F, typename... A>
    void run_parallel(const F &task, int n, const A &...args) {
        for(int i = 0; i < n; ++i) {
            for(int j = 0; j < this->discriminators[0]->num_rams; ++j)
                this->pool->push_task(std::bind(task, j, i, this, args...));
        };
        this->pool->wait_for_tasks();
    };   

    static void train_work(int ram, int i, WiSARD3* obj, ArrayND<bool>& input, ArrayND<int>& classes) {
        obj->discriminators[0]->rams[ram]->train(input, i, classes[i]);
    };

    virtual void train(ArrayND<bool>& input, ArrayND<int>& classes) {
        this->run_parallel(WiSARD3::train_work, input.shape[0], ref(input), ref(classes));
    };

    static void predict_work(int ram, int i, WiSARD3* obj, ArrayND<bool>& input, ArrayND<atomic<int>>& output, int begin, int end) {
        obj->discriminators[0]->rams[ram]->predict(input, output, i);
    };

    virtual ArrayND<atomic<int>> predict(ArrayND<bool>& input, int begin=0, int end=-1) {
        if (end == -1) end = input.shape[0];
        ArrayND<atomic<int>> output({input.shape[0], this->num_classes});
        this->run_parallel(WiSARD3::predict_work, input.shape[0], ref(input), ref(output), begin, end);
        return output;
    };

    virtual ~WiSARD3() {
        for(int i = 0; i < 1; ++i) {
            delete this->discriminators[i];
        };
        delete [] this->discriminators;
        delete this->pool;
    };

};