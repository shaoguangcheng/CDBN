#include "test.h"

////////////////// test net //////////////////
testNet::testNet(const string &configFile)
    : n(configFile)
{
}

void testNet::print() const
{
    cout << n << endl;
}

////////////////// test option ////////////////
testOption::testOption(const string &configFile)
    : opt(configFile)
{
}

void testOption::print() const
{
    cout << opt << endl;
}
