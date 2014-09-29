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

////////////////// test CRBM //////////////////
void testCRBMModel()
{
    const int DIM = 2;
    Array<double, DIM+1> W(2,2,2);
    Array<double, 1> biasV(3);
    Array<double, 1> biasH(3);
    Array<double, DIM+1> top(2,2,2);

    W = 1,2,3,4,5,6,7,8;
    biasV = 0.1,0.2,0.3;
    biasH = 0.4,0.5,0.6;
    top = 2,3,4,5,6,7,8,9;

    CRBMModel<double, DIM> crbm(W, biasV, biasH);
    crbm.writeToFile("../data/crbm");

    CRBMModel<double, DIM> _crbm_;
    _crbm_.loadFromFile("../data/crbm");
    // cout << _crbm_ << endl;

    CDBNModel<double, DIM> cdbn;
    cdbn.addCRBM(crbm);
    cdbn.addCRBM(_crbm_);
    cdbn.writeToFile("../data/cdbn");

    CDBNModel<double, DIM> _cdbn_;
    _cdbn_.loadFromFile("../data/cdbn");
    cout << _cdbn_.size() << endl;
}

///////////////////////test util /////////////////
void testAddNumber()
{
    Array<double, 2> x(3,3);
    x = 1,2,3,4,5,6,7,8,9;
    cout << addNumber(x, 2.0) << endl;
}
