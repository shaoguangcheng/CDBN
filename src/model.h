#ifndef MODEL_H
#define MODEL_H

#include <blitz/array.h>

#include "base.h"

#ifdef BZ_NAMESPACES
using namespace blitz;
#endif

/**
 * @brief CRBMModel define the structure of CRBM model which manages the weight, the bias of visible layer,
 *          the bias of hidden layer, the output of CRBM and the label(if the model is supervised)
 */
template <class T, int DIM = 2>
class CRBMModel{
public :
    CRBMModel(){}
    CRBMModel(const Array<T, DIM+1>& W,
              const Array<T, 1>& biasV,
              const Array<T, 1>& biasH,
              const Array<T, Dim+1>& top,
              const Array<T, 1>& label = -1)
        : W(W), biasV(biasV), biasH(biasH), top(top), label(label){}

    CRBMModel(const CRBMModel& m)
        : W(m.W), biasV(m.biasV), biasH(m.biasH), top(m.top), label(m.label){}

    CRBMModel& operator = (const CRBMModel& m){

    }

public :
    Array<T, DIM+1> W;
    Array<T, 1> biasV;
    Array<T, 1> biasH;
    Array<T, DIM+1> top;
    Array<T, 1> label;
};

#endif // MODEL_H
