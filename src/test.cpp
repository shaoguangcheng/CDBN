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
void testMultByElementSum()
{
    Array<double, 2> x(2,2);
    x = 1,2,3,4;
    cout << multByElementSum(x, x) << endl;
}

void testAddNumber()
{
    Array<double, 2> x(3,3);
    x = 1,2,3,4,5,6,7,8,9;
    cout << addScalar(x, 2.0) << endl;
}

void testAddVectorByRow()
{
    Array<double, 2> x(3,3);
    x = 1,2,3,4,5,6,7,8,9;

    Array<double, 1> y(3);
    y = 1,2,3;

    cout << "x : " << x << endl;
    cout << "y : " << y << endl;
    cout << addVectorByRow(x,y) << endl;
}

void testAddVectorByCol()
{
    Array<double, 2> x(3,3);
    x = 1,2,3,4,5,6,7,8,9;

    Array<double, 1> y(3);
    y = 1,2,3;

    cout << "x : " << x << endl;
    cout << "y : " << y << endl;
    cout << addVectorByCol(x,y) << endl;
}

void testMultVector()
{
    Array<double, 2> x(2,3);
    x = 1,2,3,4,5,6;

    Array<double, 1> y(3);
    y = 1,2,1;

    cout << "x : " << x << endl;
    cout << "y : " << y << endl;
    cout << multVector(x,y) << endl;
}

void testMultMatrix()
{
    Array<double, 2> x(2,3);
    x = 1,2,3,4,5,6;

    Array<double, 2> y(3,2);
    y = x;

    cout << "x : " << x << endl;
    cout << "y : " << y << endl;
    cout << multMatrix(x,y) << endl;
}

//////////////////////////test convolution /////////////////
void testConvolution2D()
{
    Array<double, 2> x(3,3);
    x = 1,2,3,4,5,6,7,8,9;

    Array<double, 2> y(3,3);
    y = 1,2,1,1,1,1,1,2,1;

    cout << "x : " << x << endl;
    cout << "y : " << y << endl;
    cout << "convolution : " << convolve(x, y, "valid") << endl;
}

void testConvolution3D()
{
    Array<double, 3> x(2,2,2);
    x = 1,2,3,4,5,6,7,8;

    Array<double, 3> y(2,2,2);
    y = 1,2,1,1,1,1,1,2;

    cout << "x : " << x << endl;
    cout << "y : " << y << endl;
    cout << "convolution : " << convolve(x, y, "valid") << endl;
}

////////////////////// test special functions //////////////
void testSigmod()
{
    Array<double, 3> x(2,2,2);
    x = 1,2,3,4,5,6,7,8;

    cout <<"sigmod : " << sigmod(x) << endl;
}

void testRandn()
{
    Array<double, 2> x(10,1);
    Array<double, 2> y(1,10);
    randn(x);
    cout << x << endl;
    y = transpose(x) ;
    cout << y << endl;
}

/////////////////////// test util ///////////////////////
void testStochasticPooling()
{
    int scale = 2;

    Array<double, 2> P(6,6);
    Array<double, 2> state(6,6);
    Array<double, 2> outPooling(6/scale,6/scale);

    P = randn(P);
    P = 0.3;
    cout << "P : " << P << endl;

    stochasticPooling(P, state, outPooling, scale);

    cout << "state : " << state << endl;
    cout << "out pooling : " << outPooling << endl;
}
