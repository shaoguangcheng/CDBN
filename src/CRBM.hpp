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

    double momentum = 0;

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
    Array<T, DIM+1> WIncTmp(shapeW);
    Array<T, 1> biasVIncTmp(nVisible);
    Array<T, 1> biasHIncTmp(convL.nFeatureMap);
    Array<T, 1> dhbias(convL.nFeatureMap);
    Array<T, DIM+1> dW(shapeW);
    Array<T, 1> dBiasV(nVisible);
    Array<T, 1> dBiasH(convL.nFeatureMap);

    /// initialize parameter
    W = randn(W);
    biasV = 0;
    biasH = 1;
    biasH = multScalar(biasH, -0.1);

    /// initialize the increasement of parameter
    WInc = 0;
    biasVInc = 0;
    biasHInc = 0;
    WIncTmp = 0;
    biasVIncTmp = 0;
    biasHIncTmp = 0;
    dW = 0;
    dBiasV = 0;
    dBiasH = 0;

    int nCase = shapeData(size-2);/// the number of examples in each feature map
    int batchSize = opt.batchSize;
    int nBatch = ceiling(nCase, batchSize); ///< batch number

    /// initialize some temple vars
    TinyVector<int, DIM+2> shapeV, shapeH, shapePooling;
    Array<T, DIM+2> visActP;
    Array<T, DIM+2> hidActP;
    Array<T, DIM+2> hidState;
    Array<T, DIM+2> outOfPooling;
    Array<T, DIM+2> hidActPTmp;
    Array<T, DIM+2> visActPTmp;

    // for visible layer
    shapeV = shapeData;
    shapeV(DIM)   = batchSize;
    shapeV(DIM+1) = nVisible;
    visActP.resize(shapeV);
    visActPTmp.resize(shapeV);
    visActP = 0;
    visActPTmp = 0;

    // for hidden layer
    for(int i = 0; i < DIM; ++i){
        shapeH(i) = shapeData(i) - shapeW(i) + 1;
    }
    shapeH(DIM) = batchSize;
    shapeH(DIM+1) = convL.nFeatureMap;
    hidActP.resize(shapeH);
    hidState.resize(shapeH);
    hidActPTmp.resize(shapeH);
    hidActP = 0;
    hidState = 0;
    hidActPTmp = 0;

    // for output of pooling
    shapePooling = shapeH;
    for(int i = 0; i < DIM; ++i){
        shapePooling(i) = shapeH(i)/poolingL.scale;
    }
    outOfPooling.resize(shapePooling);

    /// shuffle data
    Array<T, DIM+2> cpyData(shapeData);
    cpyData = shuffleData(data);

    //calculate parameters
    T error = T(0);
    for(int i = 0; i < opt.nEpoch; ++i){

        shapeV(DIM) = opt.batchSize;
        visActP.resize(shapeV);
        visActPTmp.resize(shapeV);

        shapeH(DIM) = opt.batchSize;
        hidActP.resize(shapeH);
        hidState.resize(shapeH);
        hidActPTmp.resize(shapeH);

        for(int batch = 0; batch < nBatch; ++batch){
            if(opt.batchSize*(batch+1) > nCase){
                shapeV(DIM) = nCase - batchSize*batch;
                visActP.resize(shapeV);
                visActPTmp.resize(shapeV);

                if(2 == DIM){
                    visActP = cpyData(Range::all(), Range::all(), Range(batchSize*batch, toEnd), Range::all());
                    batchSize = visActP.shape()(2);
                }
                else{
                    if(3 == DIM){
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
               hidActPTmp.resize(shapeH);
               hidState.resize(shapeH);
            }
            else{
                if(2 == DIM){
                    visActP = cpyData(Range::all(), Range::all(), Range(batchSize*batch, batchSize*(batch+1)), Range::all());
                    batchSize = visActP.shape()(2);
                }
                else{
                    if(3 == DIM){
                        visActP = cpyData(Range::all(), Range::all(), Range::all(), Range(batchSize*batch, batchSize*(batch+1)), Range::all());
                        batchSize = visActP.shape()(3);
                    }
                    else{
                        DEBUGMSG("Unsupport operation");
                        exit(EXIT_FAILURE);
                    }
                }
            }


            visActPTmp = visActP.copy();
            hidActP = inference(visActP, W, biasH);
            pooling(hidActP, hidState, outOfPooling); // do pooling
            hidActPTmp = hidActP.copy();
            computePV(hidActP, visActP, WIncTmp); // compute P(h=1|v)V for W updating
            computeP(hidActP, biasHIncTmp); // compute P(h=1|v) for bias of hidden layer updating
            computeV(visActP, biasVIncTmp); // compute V, for bias of visible layer updating

            dW = WIncTmp;
            dBiasV = biasVIncTmp;
            dBiasH = biasHIncTmp;

            // negative phase
            // contrast divergence
            for(int m = 0; m < opt.nCD; ++m){
                reconstruct(hidState, W, biasV, visActP);
                hidActP = inference(visActP, W, biasH);
                pooling(hidActP, hidState, outOfPooling);
            }

            computePV(hidActP, visActP, WIncTmp);
            computeP(hidActP, biasHIncTmp);
            computeV(visActP, biasVIncTmp);

            // compute the increasement of W, biasV, biasH
            if(!strcmp(opt.biasMode.c_str(), "none")){
                dhbias = T(0);
            }
            else{
                if(!strcmp(opt.biasMode.c_str(), "simple")){
                    computedhBias(hidActPTmp, dhbias);
                    dhbias -= opt.sparsity;
                }
                else{
                    DEBUGMSG("Undefined bias mode");
                    exit(EXIT_FAILURE);
                }
            }

            dW = (dW - WIncTmp)*1.0/batchSize - T(opt.lambda1)*W; // here fixed me
            dBiasV = (dBiasV - biasVIncTmp)*1.0/batchSize;
            dBiasH = (dBiasH - biasHIncTmp)*1.0/batchSize - opt.lambda2*dhbias;

            if(i < 5){
                momentum = initMomentum;
            }
            else{
                momentum = finalMomentum;
            }

            // update parameters
            WInc = momentum * WInc + opt.alpha * dW;
            biasVInc = momentum * biasVInc + opt.alpha * dBiasV;
            biasHInc = momentum * biasHInc + opt.alpha * dBiasH;

            W += WInc;
            biasV += biasVInc;
            biasH += biasHInc;

            // reconstruction error
            Array<T, DIM+2> errorTmp(visActPTmp-visActP);
            error += euclidNorm(errorTmp);
        }

        cout << "Epoch " << i << ", reconstruction error " << error << endl;

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
Array<T, DIM+2> CRBM<T, DIM>::inference(const Array<T, DIM+2>& batchData, const Array<T, DIM+1>& W, const Array<T, 1>& biasH)
{
    TinyVector<int, DIM+2> shapeData = batchData.shape(), shapeHidActP;
    TinyVector<int, DIM+1> shapeW = W.shape();

    int nVisible = shapeData(DIM+1), nHidden = convL.nFeatureMap;

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
                // do convolution
                hidActP(Range::all(), Range::all(), Range::all(), i) += convolve(batchData(Range::all(), Range::all(), Range::all(), j), W(Range::all(), Range::all(), i), type);
            }

            // add bias
            hidActP(Range::all(), Range::all(), Range::all(), i) = addScalar(hidActP(Range::all(), Range::all(), Range::all(), i), biasH(i));

            // if input data obeys Gaussian Distribution
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
            // for 3d case
            if(3 == DIM){
               for(int j = 0; j < nVisible; ++j){
                   // do convolution
                   hidActP(Range::all(), Range::all(), Range::all(), Range::all(), i) += convolve(batchData(Range::all(), Range::all(), Range::all(), Range::all(), j), W(Range::all(), Range::all(), Range::all(), i), type);
               }

               // add bias
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
void CRBM<T, DIM>::reconstruct(const Array<T, DIM+2>& state, const Array<T, DIM+1>& W, const Array<T, 1>& biasV, Array<T, DIM+2>& visActP)
{
    TinyVector<T, DIM+2> stateShape = state.shape();
    TinyVector<T, DIM+1> WShape = W.shape();
    TinyVector<T, DIM+2> visActPShape = visActP.shape();

    int nVisble = biasV.size(), nHidden = stateShape(DIM+1);

    visActP = 0;
    const char* type = "full";
    if(DIM == 2){
        for(int i = 0; i < nVisble; ++i){
            for(int j = 0; j < nHidden; ++j){
                /// do convolution (negative)
                visActP(Range::all(), Range::all(), Range::all(), i) += convolve(state(Range::all(), Range::all(), Range::all(), j), W(Range::all(), Range::all(), i), type);
            }

            /// add bias
            visActP(Range::all(), Range::all(), Range::all(), i) = addScalar(visActP(Range::all(), Range::all(), Range::all(), i), biasV(i));

            if(GAUSSIAN == inputType){
                // need to complete

            }
            else{
                // apply sigmod
                visActP(Range::all(), Range::all(), Range::all(), i) = sigmod(visActP(Range::all(), Range::all(), Range::all(), i));
            }
        }

        return;
    }

    if(DIM == 3){
        for(int i = 0; i < nVisble; ++i){
            for(int j = 0; j < nHidden; ++j){
                /// do convolution (ne)
                visActP(Range::all(), Range::all(), Range::all(), Range::all(), i) += convolve(state(Range::all(), Range::all(), Range::all(), Range::all(), j), W(Range::all(), Range::all(), Range::all(), i), type);
            }

            /// add bias
            visActP(Range::all(), Range::all(), Range::all(), Range::all(), i) = addScalar(visActP(Range::all(), Range::all(), Range::all(), Range::all(), i), biasV(i));

            if(GAUSSIAN == inputType){
                // need to complete
            }
            else{
                // apply sigmod
                visActP(Range::all(), Range::all(), Range::all(), Range::all(), i) = sigmod(visActP(Range::all(), Range::all(), Range::all(), Range::all(), i));
            }
        }

        return;
    }

    DEBUGMSG("Unsupport operation");
    exit(EXIT_FAILURE);
}

template <class T, int DIM>
void CRBM<T, DIM>::pooling(Array<T, DIM+2>& P, Array<T, DIM+2>& state, Array<T, DIM+2>& outPooling)
{
    TinyVector<int, DIM+2> shapeP = P.shape();
    TinyVector<int, DIM+2> shapeState = state.shape();
    TinyVector<int, DIM+2> shapeOutPooling = outPooling.shape();

#if 0
    if(shapeP != shapeState){
        DEBUGMSG("size does not match");
        exit(EXIT_FAILURE);
    }
#endif

    if(DIM == 2){
        if(MAX == poolingL.type){
            for(int i = 0; i < shapeP[DIM+1]; ++i){ // for each batch
                for(int j = 0; j < shapeP[DIM]; ++j){ // for each feature map
                    Array<T, 2> tmpP(P(Range::all(), Range::all(), j, i));
                    Array<T, 2> tmpState(state(Range::all(), Range::all(), j, i));
                    Array<T, 2> tmpOutPooling(outPooling(Range::all(), Range::all(), j, i));

                    maxPooling(tmpP, tmpState, tmpOutPooling, poolingL.scale);

                    P(Range::all(), Range::all(), j, i) = tmpP;
                    state(Range::all(), Range::all(), j, i) = tmpState;
                    outPooling(Range::all(), Range::all(), j, i) = tmpOutPooling;
                }
            }
            return;
        }

        if(MEAN == poolingL.type){
            DEBUGMSG("This pooling type has not been implemented");
            exit(EXIT_FAILURE);
        }

        if(STOCHASTIC == poolingL.type){
            for(int i = 0; i < shapeP[DIM+1]; ++i){ // for each batch
                for(int j = 0; j < shapeP[DIM]; ++j){ // for each feature map
                    Array<T, 2> tmpP(P(Range::all(), Range::all(), j, i));
                    Array<T, 2> tmpState(state(Range::all(), Range::all(), j, i));
                    Array<T, 2> tmpOutPooling(outPooling(Range::all(), Range::all(), j, i));

                    stochasticPooling(tmpP, tmpState, tmpOutPooling, poolingL.scale);

                    P(Range::all(), Range::all(), j, i) = tmpP;
                    state(Range::all(), Range::all(), j, i) = tmpState;
                    outPooling(Range::all(), Range::all(), j, i) = tmpOutPooling;
                }
            }
            return;
        }
    }

    if(DIM == 3){
        if(MAX == poolingL.type){
            for(int i = 0; i < shapeP[DIM+1]; ++i){ // for each batch
                for(int j = 0; j < shapeP[DIM]; ++j){ // for each feature map
                    Array<T, 3> tmpP(P(Range::all(), Range::all(), Range::all(), j, i));
                    Array<T, 3> tmpState(state(Range::all(), Range::all(), Range::all(), j, i));
                    Array<T, 3> tmpOutPooling(outPooling(Range::all(), Range::all(), Range::all(), j, i));

                    maxPooling(tmpP, tmpState, tmpOutPooling, poolingL.scale);

                    P(Range::all(), Range::all(), Range::all(), j, i) = tmpP;
                    state(Range::all(), Range::all(), Range::all(), j, i) = tmpState;
                    outPooling(Range::all(), Range::all(), Range::all(), j, i) = tmpOutPooling;
                }
            }
            return;
        }

        if(MEAN == poolingL.type){
            DEBUGMSG("This pooling type has not been implemented");
            exit(EXIT_FAILURE);
        }

        if(STOCHASTIC == poolingL.type){
            for(int i = 0; i < shapeP[DIM+1]; ++i){ // for each batch
                for(int j = 0; j < shapeP[DIM]; ++j){ // for each feature map
                    Array<T, 3> tmpP(P(Range::all(), Range::all(), Range::all(), j, i));
                    Array<T, 3> tmpState(state(Range::all(), Range::all(), Range::all(), j, i));
                    Array<T, 3> tmpOutPooling(outPooling(Range::all(), Range::all(), Range::all(), j, i));

                    stochasticPooling(tmpP, tmpState, tmpOutPooling, poolingL.scale);

                    P(Range::all(), Range::all(), Range::all(), j, i) = tmpP;
                    state(Range::all(), Range::all(), Range::all(), j, i) = tmpState;
                    outPooling(Range::all(), Range::all(), Range::all(), j, i) = tmpOutPooling;
                }
            }
            return;
        }
    }

    DEBUGMSG("Unsupport pooling type");
    exit(EXIT_FAILURE);
}

template <class T, int DIM>
void CRBM<T, DIM>::computePV(const Array<T, DIM+2>& hidActP, const Array<T, DIM+2>& visActP, Array<T, DIM+1>& PV)
{
    TinyVector<int, DIM+2> shapeHidActP = hidActP.shape();
    TinyVector<int, DIM+2> shapeVisActP = visActP.shape();
    TinyVector<int, DIM+1> shapePV = PV.shape();

    int nVisble = shapeVisActP[DIM+1], nHidden = shapeHidActP[DIM+1], nCase = shapeVisActP[DIM];

    PV = 0;
    const char* type = "valid";
    if(DIM == 2){
        for(int i = 0; i < nHidden; ++i){
            /// for each convolutional kernel
            for(int k = 0; k < nCase; ++k){
                /// for each example
                for(int j = 0; j < nVisble; ++j){
                    PV(Range::all(), Range::all(), i) += convolve(visActP(Range::all(), Range::all(), k, j), hidActP(Range::all(), Range::all(), k, i), type);
                }
            }
        }

        int numcases = shapeHidActP(0)*shapeHidActP(1)*nVisble;
        PV = divideScalar(PV, T(numcases));

        return;
    }

    if(DIM == 3){
        for(int i = 0; i < nHidden; ++i){
            /// for each convolutional kernel
            for(int k = 0; k < nCase; ++k){
                /// for each example
                for(int j = 0; j < nVisble; ++j){
                    PV(Range::all(), Range::all(), Range::all(), i) += convolve(visActP(Range::all(), Range::all(), Range::all(), k, j), hidActP(Range::all(), Range::all(), Range::all(), k, i), type);
                }
            }
        }

        int numcases = shapeHidActP[0]*shapeHidActP[1]*shapeHidActP[2]*nVisble;
        PV = divideScalar(PV, T(numcases));

        return;
    }

    DEBUGMSG("Unsupport operation");
    exit(EXIT_FAILURE);
}

template <class T, int DIM>
void CRBM<T, DIM>::computeP(const Array<T, DIM+2>& hidActP, Array<T, 1>& biasHInc)
{
    TinyVector<int, DIM+2> shapeHidActP = hidActP.shape();

    int nHidden = biasHInc.size();

    biasHInc = 0;

    if(DIM == 2){
        for(int i = 0; i < nHidden; ++i){
            biasHInc(i) = sum(hidActP(Range::all(), Range::all(), Range::all(), i));
            int numcases = shapeHidActP(0)*shapeHidActP(1);
            biasHInc(i) /= numcases;
        }
        return;
    }

    if(DIM == 3){
        for(int i = 0; i < nHidden; ++i){
            biasHInc(i) = sum(hidActP(Range::all(), Range::all(), Range::all(), Range::all(), i));
            int numcases = shapeHidActP(0)*shapeHidActP(1)*shapeHidActP(2);
            biasHInc(i) /= numcases;
        }
        return;
    }

    DEBUGMSG("Unsupport operation");
    exit(EXIT_FAILURE);
}

template <class T, int DIM>
void CRBM<T, DIM>::computeV(const Array<T, DIM+2>& visActP, Array<T, 1>& biasVInc)
{
    TinyVector<int, DIM+2> shapeVisActP = visActP.shape();

    int nVisble = biasVInc.size();

    biasVInc = 0;

    if(DIM == 2){
        for(int i = 0; i < nVisble; ++i){
            biasVInc(i) = sum(visActP(Range::all(), Range::all(), Range::all(), i));
            int numcases = shapeVisActP(0)*shapeVisActP(1);
            biasVInc(i) /= numcases;
        }

        return;
    }

    if(DIM == 3){
        for(int i = 0; i < nVisble; ++i){
            biasVInc(i) = sum(visActP(Range::all(), Range::all(), Range::all(), Range::all(), i));
            int numcases = shapeVisActP(0)*shapeVisActP(1)*shapeVisActP(2);
            biasVInc(i) /= numcases;
        }

        return;
    }

    DEBUGMSG("Unsupport operation");
    exit(EXIT_FAILURE);
}

template <class T, int DIM>
void CRBM<T, DIM>::computedhBias(const Array<T, DIM+2> &hidActP, Array<T, 1>& dhbias)
{
    TinyVector<int, DIM+2> shapeHidActP = hidActP.shape();
    int nHidden = dhbias.size();

    dhbias = T(0);

    if(DIM == 2){
        for(int i = 0; i < nHidden; ++i){
            dhbias(i) = mean(hidActP(Range::all(), Range::all(), Range::all(), i));
        }

        return;
    }

    if(DIM == 3){
        for(int i = 0; i < nHidden; ++i){
            dhbias(i) = mean(hidActP(Range::all(), Range::all(), Range::all(), Range::all(), i));
        }

        return;
    }

    DEBUGMSG("Unsupport operation");
    exit(EXIT_FAILURE);
}

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
