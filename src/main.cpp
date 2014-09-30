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

    testCRBM crbm;

//    testMultByElementSum();
//    testAddNumber();
//    testAddVectorByRow();
//    testAddVectorByCol();
//    testMultVector();
//    testMultMatrix();
//    testConvolution2D();
    testConvolution3D();

    return 0;
}

