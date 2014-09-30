#ifndef TEST_H
#define TEST_H

#include <iostream>

#include <blitz/array.h>

#include "base.h"
#include "model.h"
#include "CRBM.h"
#include "matrixOperation.h"

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

/////////////////// test crbmModel ///////////////////
void testCRBMModel();

/////////////////// test CRBM ///////////////////////
class testCRBM
{
public :
    testCRBM(){}

private :
    CRBM<double, 2> crbm;
};

////////////////////// test util ////////////////////
void testMultByElementSum();
void testAddNumber();
void testAddVectorByRow();
void testAddVectorByCol();
void testMultVector();
void testMultMatrix();

////////////////////// test convolution ////////////////
void testConvolution2D();
void testConvolution3D();

////////////////////// test special functions /////////////
void testSigmod();
void testRandn();

////////////////////// test CRBM ///////////////////////

#endif // TEST_H
