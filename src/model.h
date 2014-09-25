#ifndef MODEL_H
#define MODEL_H

#include <blitz/array.h>
//#include <blitz/vector-et.h>

#include <vector>
#include <cstddef>
#include <fstream>

#include "global.h"

using namespace std;

#ifdef BZ_NAMESPACES
using namespace blitz;
#endif

template <class T, int DIM> class CRBMModel;

template <class T, int DIM>
ostream& operator << (ostream& out, const CRBMModel<T, DIM>& crbm);
template <class T, int DIM>
istream& operator >> (istream& in, CRBMModel<T, DIM>& crbm);

/**
 * @brief CRBMModel define the structure of CRBM model which manages the weight, the bias of visible layer,
 *          the bias of hidden layer, the output of CRBM and the label (if the model is supervised)
 */
template <class T, int DIM>
class CRBMModel{
public :
    CRBMModel(){}
    CRBMModel(const Array<T, DIM+1>& W,
              const Array<T, 1>& biasV,
              const Array<T, 1>& biasH,
              const Array<T, DIM+1>& top)
        : W(W), biasV(biasV), biasH(biasH), top(top){}

    CRBMModel(const CRBMModel& m)
        : W(m.W), biasV(m.biasV), biasH(m.biasH), top(m.top){}

    CRBMModel& operator = (const CRBMModel& m){
        W     = m.W;
        biasV = m.biasV;
        biasH = m.biasH;
        top   = m.top;

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

    friend ostream& operator << <T, DIM>(ostream& out, const CRBMModel<T, DIM>& crbm);
    friend istream& operator >> <T, DIM>(istream& in,  CRBMModel<T, DIM>& crbm);

public :
    Array<T, DIM+1> W;
    Array<T, 1> biasV;
    Array<T, 1> biasH;
    Array<T, DIM+1> top;
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
    std::vector<CRBMModel<T, DIM> > model;
};

#include "model.hpp" // Which way decides your compiler

#endif // MODEL_H
