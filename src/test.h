#ifndef TEST_H
#define TEST_H

#include <iostream>

#include <blitz/array.h>

#include "base.h"
#include "model.h"

using namespace std;

using namespace blitz;

//////////////////// test net ///////////////////
class testNet
{
public:
    testNet(){}
    testNet(const string& configFile);

    void print() const;

private :
    net n;
};

/////////////////// test option //////////////////
class testOption{
public :
    testOption(){}
    testOption(const string& configFile);

    void print() const;

private :
    option opt;
};

/////////////////// test crbm ///////////////////
void testCRBM();

#endif // TEST_H
