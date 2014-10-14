#ifndef CRBM_H
#define CRBM_H

#include <blitz/vector-et.h>

#include "base.h"
#include "model.h"
#include "matrixOperation.h"
#include "util.h"

template <class T, int DIM>
class CRBM{
    /**
     * @brief initMomentum
     * @brief finalMomentum these two variables are always fixed
     */
    static double initMomentum;
    static double finalMomentum;

public :
    CRBM(){}
    CRBM(const Array<T, DIM+2>& data, const convLayer& convL, const poolingLayer& poolingL, const option& opt, layerType inputType);
    CRBM(const CRBM<T, DIM>& crbm);
    CRBM& operator = (const CRBM<T, DIM>& crbm);

    /**
     * @brief train train parameters for CRBM
     */
    void train();

    /**
     * @brief train train CRBM parameters for online update
     * @param crbm original CRBM model
     */
    void train(const CRBMModel<T, DIM>& crbm);

    /**
     * @brief getModel get the trained model
     * @return trained model
     */
    inline const CRBMModel<T, DIM>& getModel() const {
        return crbmM;
    }

private :
    /**
     * @brief feedForward compute the ouput according to the input data. crbmM.top will be filled after
     *        this process
     */
    void feedForward();

    /**
     * @brief inference from visible layer to hidden layer
     * @param batchData
     * @param W
     * @param biasH
     * @return
     */
    Array<T, DIM+2> inference(const Array<T, DIM+2>& batchData, const Array<T, DIM+1>& W, const Array<T, 1> &biasH);

    /**
     * @brief reconstruct from hidden layer to visible layer
     * @param state
     * @param W
     * @param biasV
     * @param visActP
     */
    void reconstruct(const Array<T, DIM+2>& state, const Array<T, DIM+1>& W, const Array<T, 1>& biasV, Array<T, DIM+2>& visActP);

    /**
     * @brief pooling do pooling for hidden layer
     * @param P
     * @param state
     * @param outPooling
     */
    void pooling(Array<T, DIM+2>& P, Array<T, DIM+2> &state, Array<T, DIM+2>& outPooling);

    /**
     * @brief computePV compute P(h=1|v)V
     * @param hidActP
     * @param visActP
     * @param PV the result of  P(h=1|v)V
     */
    void computePV(const Array<T, DIM+2>& hidActP, const Array<T, DIM+2>& visActP, Array<T, DIM+1>& PV);

    /**
     * @brief computeP compute P(h=1|v), for bias of hidden layer updating
     * @param hidActP
     * @param biasHInc result of P(h=1|v)
     */
    void computeP(const Array<T, DIM+2>& hidActP, Array<T, 1>& biasHInc);

    /**
     * @brief computeV compute V for bias of visible layer updating
     * @param visActP
     * @param biasVInc result of V
     */
    void computeV(const Array<T, DIM+2>& visActP, Array<T, 1>& biasVInc);

    /**
     * @brief computedhBias compute dhbias for parameters updating
     * @param hidActP
     * @param dhbias result of dhbias
     */
    void computedhBias(const Array<T, DIM+2> &hidActP, Array<T, 1>& dhbias);

private :
    void trimDataForPooling(Array<T, DIM+2>& batchData, int kernelSize, int blockSize);

private :
    /**
     * @brief data store input data of CRBM
     */
    Array<T, DIM+2> data;

    // A CRBM unit contains a convolution layer and pooling layer
    convLayer convL;
    poolingLayer poolingL;

    /**
     * @brief isInputGaussian  is input data from Gaussian distribution
     */
    layerType inputType;

    /**
     * @brief opt store miscellanous parameters
     */
    option opt;

    /**
     * @brief crbmM save the parameters of CRBM model
     */
    CRBMModel<T, DIM> crbmM;
};

#include "CRBM.hpp"

#endif // CRBM_H
