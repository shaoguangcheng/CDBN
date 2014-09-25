#ifndef TEST_H
#define TEST_H

#include <iostream>

#include "base.h"

using namespace std;

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

#endif // TEST_H
