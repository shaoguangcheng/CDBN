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
    CRBM(const Array<T, DIM+2>& data, const convLayer& convL, const poolingLayer& poolingL, const option& opt, bool isInputGaussian = false);
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
     */
    Array<T, DIM+2> inference(const Array<T, DIM+2>& batchData, const Array<T, DIM+1>& W, const Array<T, 1> biasH);

    /**
     * @brief reconstruct from hidden layer to visible layer
     */
    void reconstruct();

    /**
     * @brief pooling do pooling for hidden layer
     */
    void pooling();

private :
    void copyData();

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
    bool isInputGaussian;

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
