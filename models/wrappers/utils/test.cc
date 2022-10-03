#include <iostream>
#include <vector>
#include <atomic>

using namespace std;

int main(void) {

    int t[4] = {5, 3, 4, 2};

    vector<ssize_t> t2(t, t+4);

    for(int i = 0; i < 4; ++i)
        cout << t2[i] << " ";
    cout << endl;


    return 0;
};