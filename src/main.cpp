#include <iostream>

#include "test.h"
#include "util.h"

using namespace std;

int main()
{
    testNet tn("../config/net.cfg");
    tn.print();

    testOption opt("../config/option.cfg");
    opt.print();

    testCRBMModel();

//    testCRBM crbm;

//    testMultByElementSum();
//    testAddNumber();
//    testAddVectorByRow();
//    testAddVectorByCol();
//    testMultVector();
//    testMultMatrix();
//    testConvolution2D();
//    testConvolution3D();
//    testSigmod();
//    testRandn();
    Array<double,4> data(4,5,5,2);
    firstIndex i;
    secondIndex j;
    thirdIndex k;
    fourthIndex l;

    data = i+j+k+l;

    int size = data[0].shape().length();
    cout << data[0].shape()(size-1) << endl;

    convLayer convL("convolution", 3, 2, 1);
    poolingLayer poolingL("pooling", 3.0, MAX);

    CRBM<double, 2> crbm(data, convL, poolingL, option("../config/option.cfg"), GAUSSIAN);

    crbm.train();

    return 0;
}

