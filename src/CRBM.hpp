#include "CRBM.h"

template <class T, int DIM>
double CRBM<T, DIM>::initMomentum = 0.5;
template <class T, int DIM>
double CRBM<T, DIM>::finalMomentum = 0.9;

template <class T, int DIM>
CRBM<T, DIM>::CRBM(const std::vector<Array<T, DIM+1> > &data,
                   const convLayer &convL,
                   const poolingLayer &poolingL,
                   const option &opt)
    : data(data), convL(convL), poolingL(poolingL), opt(opt)
{
    copyData();
}

template <class T, int DIM>
CRBM<T, DIM>::CRBM(const CRBM<T, DIM> &crbm)
    : data(crbm.data), convL(crbm.convL), poolingL(crbm.poolingL), opt(crbm.opt)
{
    copyData();
}

template <class T, int DIM>
CRBM<T, DIM>& CRBM<T, DIM>::operator =(const CRBM<T, DIM>& crbm)
{
    data = crbm.data;
    convL = crbm.convL;
    poolingL = crbm.poolingL;
    opt = crbm.opt;

    copyData();

    return *this;
}

template <class T, int DIM>
void CRBM<T, DIM>::copyData()
{
    size_t size = data.size();
    Array<T, DIM+1> tmp;
    for(int i=0; i<size; ++i){
        tmp = data[i].copy();
        cpyData.push_back(tmp);
    }
}

template <class T, int DIM>
void CRBM<T, DIM>::train()
{
    int nVisible = cpyData.size(); ///< feature map number of visible layer

    TinyVector<int, DIM+1> shape;
    for(int i = 0; i < DIM;++i)
        shape(i) = convL.kernelSize;
    shape(DIM) = convL.nFeatureMap;

    Array<T, DIM+1> W(shape);
    Array<T, 1> biasV(nVisible);
    Array<T, 1> biasH(convL.nFeatureMap);

    Array<T, DIM+1> WInc(shape);
    Array<T, 1> biasVInc(nVisible);
    Array<T, 1> biasHInc(convL.nFeatureMap);

    /// initialize parameter
    W = randn(W);
    biasV = 0;
    biasH = 1;
    biasH = multScalar(biasH, -0.1);

    /// initialize the increasment of parameter
    WInc = 0;
    biasVInc = 0;
    biasHInc = 0;


}

template <class T, int DIM>
void CRBM<T, DIM>::inference()
{}

template <class T, int DIM>
void CRBM<T, DIM>::reconstruct()
{}

template <class T, int DIM>
void CRBM<T, DIM>::pooling()
{}

template <class T, int DIM>
void CRBM<T, DIM>::feedForward()
{}
