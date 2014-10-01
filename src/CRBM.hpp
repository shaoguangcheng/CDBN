#include "CRBM.h"

template <class T, int DIM>
double CRBM<T, DIM>::initMomentum = 0.5;
template <class T, int DIM>
double CRBM<T, DIM>::finalMomentum = 0.9;

template <class T, int DIM>
CRBM<T, DIM>::CRBM(const Array<T, DIM+2>&data,
                   const convLayer &convL,
                   const poolingLayer &poolingL,
                   const option &opt,
                   bool isInputGaussian)
    : data(data), convL(convL), poolingL(poolingL), opt(opt), isInputGaussian(isInputGaussian)
{
}

template <class T, int DIM>
CRBM<T, DIM>::CRBM(const CRBM<T, DIM> &crbm)
    : data(crbm.data), convL(crbm.convL), poolingL(crbm.poolingL), opt(crbm.opt), isInputGaussian(crbm.isInputGaussian)
{
}

template <class T, int DIM>
CRBM<T, DIM>& CRBM<T, DIM>::operator =(const CRBM<T, DIM>& crbm)
{
    data = crbm.data;
    convL = crbm.convL;
    poolingL = crbm.poolingL;
    opt = crbm.opt;
    isInputGaussian = crbm.isInputGaussian;

    return *this;
}

template <class T, int DIM>
void CRBM<T, DIM>::train()
{
    TinyVector<int, DIM+2> shapeData = data.shape(); ///< feature map number of visible layer
    int size = shapeData.length();

    TinyVector<int, DIM+1> shapeW;
    for(int i = 0; i < DIM;++i)
        shapeW(i) = convL.kernelSize;
    shapeW(DIM) = convL.nFeatureMap;
    int nVisible = shapeData(size-1);

    Array<T, DIM+1> W(shapeW);
    Array<T, 1> biasV(nVisible);
    Array<T, 1> biasH(convL.nFeatureMap);

    Array<T, DIM+1> WInc(shapeW);
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

    int nCase = shapeData(size-2);/// the number of examples in each feature map
    int batchSize = opt.batchSize;
    int nBatch = ceiling(nCase, batchSize);

    /// initialize some temple vars
    TinyVector<int, DIM+2> shape;
    for(int i = 0; i < DIM; ++i){
        shape(i) = shapeData(i);
    }
    shape(DIM)   = batchSize;
    shape(DIM+1) = nVisible;
    Array<T, DIM+2> visActP(shape);

    shape(DIM+1) = convL.nFeatureMap;
    Array<T, DIM+2> hidActP(shape);
    Array<T, DIM+2> hidState(shape);

    /// shuffle data
    Array<T, DIM+2> cpyData(shapeData);
    cpyData = shuffleData(data);

    //calculate parameters
    for(int i = 0; i < opt.nEpoch; ++i){
        Array<double, 1> error(opt.nEpoch);

        for(int batch = 0; batch < nBatch; ++batch){
            if(opt.batchSize*(batch+1) > nCase){
                shape(DIM) = nCase - batchSize*batch;
                shape(DIM+1) = nVisible;
                visActP.resize(shape);

                if(4 == size){
                    visActP = cpyData(Range::all(), Range::all(), Range(batchSize*batch, toEnd), Range::all());
                    batchSize = visActP.shape()(2);
                }
                else{
                    if(5 == size){
                        visActP = cpyData(Range::all(), Range::all(), Range::all(), Range(batchSize*batch, toEnd), Range::all());
                        batchSize = visActP.shape()(3);
                    }
                    else{
                        DEBUGMSG("Unsupport operation");
                        exit(EXIT_FAILURE);
                    }
                }
            }
            else{
                if(4 == size){
                    visActP = cpyData(Range::all(), Range::all(), Range(batchSize*batch, batchSize*(batch+1)), Range::all());
                    batchSize = visActP.shape()(2);
                }
                else{
                    if(5 == size){
                        visActP = cpyData(Range::all(), Range::all(), Range::all(), Range(batchSize*batch, batchSize*(batch+1)), Range::all());
                        batchSize = visActP.shape()(3);
                    }
                    else{
                        DEBUGMSG("Unsupport operation");
                        exit(EXIT_FAILURE);
                    }
                }
            }
            cout << visActP << endl;
 //           hidActP = inference(visActP, W, biasH);
        }

    }
}

template <class T, int DIM>
Array<T, DIM+2> CRBM<T, DIM>::inference(const Array<T, DIM+2>& batchData, const Array<T, DIM+1>& W, const Array<T, 1> biasH)
{
    TinyVector<int, DIM+2> shape = batchData.shape();
    int size = shape.length(), nVisible = shape(size-1), nHidden = convL.nFeatureMap;

    shape(DIM+1) = nHidden;
    Array<T, DIM+2> hidActP(shape);

    TinyVector<int, DIM+1> tmpShape;
    for(int i = 0; i < DIM+1; ++i)
        tmpShape(i) = shape(i);

    Array<T, DIM+1> tmp(tmpShape);
    for(int i = 0; i < nHidden; ++i){
        tmp = 0;
        for(int j = 0;j < nVisible; ++j){
            if(4 == size){ /// for 2d case
                tmp = tmp + convolve(batchData(Range::all(), Range::all(), Range::all(), j), W(Range::all(), Range::all(), i), "valid");
            }
            else{
                if(5 == size){ /// for 3d case
                    tmp  = tmp + convolve(batchData(Range::all(), Range::all(), Range::all(), j), W(Range::all(), Range::all(), Range::all(),i), "valid");
                }
                else{
                    DEBUGMSG("Unsupport operation");
                    exit(EXIT_FAILURE);
                }
            }
        }

        tmp = addScalar(tmp, biasH(i));
        if(4 == size){ /// for 2d case
            hidActP(Range::all(), Range::all(), Range::all(), i) = tmp;
        }
        else{
            if(5 == size){ /// for 3d case
                hidActP(Range::all(), Range::all(), Range::all(), Range::all(), i) = tmp;
            }
            else{
                DEBUGMSG("Unsupport operation");
                exit(EXIT_FAILURE);
            }
        }

    }

    return hidActP;
}

template <class T, int DIM>
void CRBM<T, DIM>::reconstruct()
{}

template <class T, int DIM>
void CRBM<T, DIM>::pooling()
{}

template <class T, int DIM>
void CRBM<T, DIM>::feedForward()
{}
