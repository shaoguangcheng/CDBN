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
    int nVisible = cpyData.size();
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
