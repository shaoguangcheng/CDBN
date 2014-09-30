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
    vector<Array<double,3> > data(3);
    convLayer convL("convolution", 3, 5, 1);
    poolingLayer poolingL("pooling", 2.0, MAX);
    for(int i=0;i<data.size();++i)
        data[i] = 1;

    CRBM<double, 2> crbm(data, convL, poolingL, option("../config/option.cfg"));

    crbm.train();

    return 0;
}

