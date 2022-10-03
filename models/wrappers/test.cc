#include <iostream>
#include <bitset>

using namespace std;

int main() {

    cout << (uint64_t)0 - 1 << endl;
    cout << bitset<64>((uint64_t)0 - 1) << endl;
    cout << bitset<64>((1 << 13) - 1) << endl;
    cout << 1 / 0.33 << endl;
    cout << (0.00001 == 0.00001) << endl;

    return 0;
};