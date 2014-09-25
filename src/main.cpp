#include <iostream>

#include "test.h"

using namespace std;

int main()
{
    testNet tn("../config/net.cfg");
    tn.print();

    testOption opt("../config/option.cfg");
    opt.print();

    return 0;
}

