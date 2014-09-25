#ifndef MODEL_H
#define MODEL_H

#include <blitz/array.h>

#include <vector>
#include <cstddef>

using namespace std;

#ifdef BZ_NAMESPACES
using namespace blitz;
#endif

/**
 * @brief CRBMModel define the structure of CRBM model which manages the weight, the bias of visible layer,
 *          the bias of hidden layer, the output of CRBM and the label(if the model is supervised)
 */
template <class T, int DIM>
class CRBMModel{
public :
    CRBMModel(){}
    CRBMModel(const Array<T, DIM+1>& W,
              const Array<T, 1>& biasV,
              const Array<T, 1>& biasH,
              const Array<T, DIM+1>& top,
              const Array<T, 1>& label = -1)
        : W(W), biasV(biasV), biasH(biasH), top(top), label(label){}

    CRBMModel(const CRBMModel& m)
        : W(m.W), biasV(m.biasV), biasH(m.biasH), top(m.top), label(m.label){}

    CRBMModel& operator = (const CRBMModel& m){
        W     = m.W;
        biasV = m.biasV;
        biasH = m.biasH;
        top   = m.top;
        label = m.label;

        return *this;
    }

    /**
     * @brief writeToFile write CRBM model to a specified file
     * @param fileName
     */
    void writeToFile(const string& fileName) const;

    /**
     * @brief loadFromFile load the CRBM model from a specified file
     * @param filename
     */
    void loadFromFile(const string& filename);

public :
    Array<T, DIM+1> W;
    Array<T, 1> biasV;
    Array<T, 1> biasH;
    Array<T, DIM+1> top;
    Array<T, 1> label;
};

template <class T, int DIM>
class CDBNModel
{
public :
    CDBNModel(){}

    inline size_t size() const {
        return model.size();
    }

    inline void addCRBM(const CRBMModel<T, DIM>& crbm) {
        model.push_back(crbm);
    }

    void writeToFile(const string& fileName) const;
    void loadFromFile(const string& fileName);

private :
    /**
     * @brief model all trained model
     */
    vector<CRBMModel<T, DIM> > model;
};

//#include "model.hpp"

#endif // MODEL_H
