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
                   layerType inputType)
    : data(data), convL(convL), poolingL(poolingL), opt(opt), inputType(inputType)
{
}

template <class T, int DIM>
CRBM<T, DIM>::CRBM(const CRBM<T, DIM> &crbm)
    : data(crbm.data), convL(crbm.convL), poolingL(crbm.poolingL), opt(crbm.opt), inputType(crbm.inputType)
{
}

template <class T, int DIM>
CRBM<T, DIM>& CRBM<T, DIM>::operator =(const CRBM<T, DIM>& crbm)
{
    data = crbm.data;
    convL = crbm.convL;
    poolingL = crbm.poolingL;
    opt = crbm.opt;
    inputType = crbm.inputType;

    return *this;
}

template <class T, int DIM>
void CRBM<T, DIM>::train()
{
    trimDataForPooling(data, convL.kernelSize, poolingL.scale);

    TinyVector<int, DIM+2> shapeData = data.shape();
    TinyVector<int, DIM+1> shapeW;

    for(int i = 0; i < DIM;++i)
        shapeW(i) = convL.kernelSize;
    shapeW(DIM) = convL.nFeatureMap;

    int size = shapeData.length();
    int nVisible = shapeData(size-1);  ///< feature map number of visible layer

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
    int nBatch = ceiling(nCase, batchSize); ///< batch number

    /// initialize some temple vars
    TinyVector<int, DIM+2> shapeV, shapeH;
    Array<T, DIM+2> visActP;
    Array<T, DIM+2> hidActP;
    Array<T, DIM+2> hidState;

    shapeV = shapeData;
    shapeV(DIM)   = batchSize;
    shapeV(DIM+1) = nVisible;
    visActP.resize(shapeV);
    visActP = 0;

    for(int i = 0; i < DIM; ++i){
        shapeH(i) = shapeData(i) - shapeW(i) + 1;
    }
    shapeH(DIM) = batchSize;
    shapeH(DIM+1) = convL.nFeatureMap;
    hidActP.resize(shapeH);
    hidState.resize(shapeH);
    hidActP = 0;
    hidState = 0;

    /// shuffle data
    Array<T, DIM+2> cpyData(shapeData);

    cpyData = shuffleData(data);

    //calculate parameters
    for(int i = 0; i < opt.nEpoch; ++i){
        //do some promopt
        cout << "Epoch : " << i << endl;

        Array<double, 1> error(opt.nEpoch);

        shapeV(DIM) = opt.batchSize;
        visActP.resize(shapeV);

        shapeH(DIM) = opt.batchSize;
        hidActP.resize(shapeH);
        hidState.resize(shapeH);

        for(int batch = 0; batch < nBatch; ++batch){
            if(opt.batchSize*(batch+1) > nCase){
                shapeV(DIM) = nCase - batchSize*batch;
                visActP.resize(shapeV);

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

               shapeH(DIM) = batchSize;
               hidActP.resize(shapeH);
               hidState.resize(shapeH);
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

            hidActP = inference(visActP, W, biasH);
        }

    }
}

/**
 * @brief inference In this step, we compute P(h|v)
 * @param batchData a batch data
 * @param W weight to optimize
 * @param biasH the bias of detection layer for each feature map
 * @return P(h|v)
 */
template <class T, int DIM>
Array<T, DIM+2> CRBM<T, DIM>::inference(const Array<T, DIM+2>& batchData, const Array<T, DIM+1>& W, const Array<T, 1> biasH)
{
    TinyVector<int, DIM+2> shapeData = batchData.shape(), shapeHidActP;
    TinyVector<int, DIM+1> shapeW = W.shape();

    int nVisible = shapeData(shapeData.length()-1), nHidden = convL.nFeatureMap;

    for(int i = 0; i < DIM; ++i){
        shapeHidActP(i) = shapeData(i) - shapeW(i) + 1;
    }
    shapeHidActP(DIM) = shapeData(DIM);
    shapeHidActP(DIM+1) = nHidden;

    Array<T, DIM+2> hidActP(shapeHidActP);
    hidActP = 0;

    const char* type = "valid";
    for(int i = 0; i < nHidden; ++i){
        /// for 2d case
        if(2 == DIM){
            for(int j = 0; j < nVisible; ++j){
                /// do convolution
                hidActP(Range::all(), Range::all(), Range::all(), i) += convolve(batchData(Range::all(), Range::all(), Range::all(), j), W(Range::all(), Range::all(), i), type);
            }

            /// add bias
            hidActP(Range::all(), Range::all(), Range::all(), i) = addScalar(hidActP(Range::all(), Range::all(), Range::all(), i), biasH(i));

            /// if input data obeys Gaussian Distribution
            if(GAUSSIAN == inputType){
                // need complete
            }
            else{
                // need complete
            }

            // here should be taken carefully, is right to use sigmod here
            hidActP(Range::all(), Range::all(), Range::all(), i) = sigmod(hidActP(Range::all(), Range::all(), Range::all(), i));
        }
        else{
            /// for 3d case
            if(3 == DIM){
               for(int j = 0; j < nVisible; ++j){
                   /// do convolution
                   hidActP(Range::all(), Range::all(), Range::all(), Range::all(), i) += convolve(batchData(Range::all(), Range::all(), Range::all(), Range::all(), j), W(Range::all(), Range::all(), Range::all(), i), type);
               }

               /// add bias
               hidActP(Range::all(), Range::all(), Range::all(), Range::all(), i) = addScalar(hidActP(Range::all(), Range::all(), Range::all(), Range::all(), i), biasH(i));

               if(GAUSSIAN == inputType){
                  // need complete
               }
               else{
                  // need complete
               }

            // here should be taken carefully, is right to use sigmod here
               hidActP(Range::all(), Range::all(), Range::all(), Range::all(), i) = sigmod(hidActP(Range::all(), Range::all(), Range::all(), Range::all(), i));
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

template <class T, int DIM>
void CRBM<T, DIM>::trimDataForPooling(Array<T, DIM+2>& batchData, int kernelSize, int blockSize)
{
    TinyVector<int, DIM+2> shape = batchData.shape();

    for(int i = 0; i < DIM; ++i){
        if((shape[i] - kernelSize + 1)%blockSize != 0){
            shape[i] = (shape[i] - kernelSize + 1)/blockSize*blockSize + kernelSize - 1;
        }
    }

    batchData.resizeAndPreserve(shape);
}
