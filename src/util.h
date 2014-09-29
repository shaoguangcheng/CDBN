#ifndef UTIL_H
#define UTIL_H

#include <blitz/array.h>
#include <blitz/array/convolve.h>
#include <blitz/vector-et.h>

#include <string.h>

#include "global.h"

#ifdef BZ_NAMESPACES
using namespace blitz;
#endif

/**
 * compute the transpose of x
 */
template <class T>
inline Array<T, 2>& transpose(const Array<T, 2>& x)
{
    return x.transpose(secondDim, firstDim);
}

template <class T>
inline Array<T,3>& transpose(const Array<T, 3>& x)
{
    return x.transpose(thirdDim, secondDim, firstDim);
}

/**
 * compute c = sumij(Xij*Yij)
 */
template<class T>
T multByElementSum(const Array<T, 2>& x, const Array<T, 2>& y)
{
    if(x.shape()(0) != y.shape()(0)||x.shape()(1) != y.shape()(1)){
        DEBUGMSG("Size can not match");
        exit(EXIT_FAILURE);
    }

    T result=(T)0;
    for(int i=0;i<x.shape()(0);++i){
        for(int j=0;j<x.shape()(1);++j){
            result += x(i,j)*y(i,j);
        }
    }

    return result;
}

/**
 * compute c = Xij+Y
 */
template <class T>
Array<T, 2> addNumber(const Array<T, 2>& x, T y)
{
    firstIndex i;
    secondIndex j;

    Array<T, 2> result(x.shape());
    result = x(i,j)+y;

    return result;
}

template<class T>
Array<T, 2> addVectorByRow(const Array<T, 2>& x, const Array<T, 1>& y)
{
    if()
}

template<class T>
Array<T, 2> addVectorByCol(const Array<T, 2>& x, )
{

}

template <class T>
Array<T, 2> convolve(const Array<T, 2>& x, const Array<T, 2>& kernel, char* type)
{
    TinyVector<int, 2> shapeX = x.shape(), shapeKernel = kernel.shape();

    const int first = 0;
    const int second = 1;

    if(!strcmp(type, "valid")){
        if(shapeKernel(first) > shapeX(first)||shapeKernel(second) > shapeX(second)){
            DEBUGMSG("In valid mode, kernel size can not be larger than data size");
            exit(EXIT_FAILURE);
        }

        Array<T, 2> result(shapeX(first)-shapeKernel(first)+1, shapeX(second)-shapeKernel(second)+1);
        for(int i=0;i+shapeKernel(first)<=shapeX(first)+1;++i){
            for(int j=0;j+shapeKernel(second)<=shapeX(second);++j){
                TinyVector<int, 2> lowerBound(i,j), upperBound(i+shapeKernel(first)-1, j+shapeKernel(second)-1);
                RectDomain<2> subDomain(lowerBound, upperBound);
                Array<T, 2> tmp = x(subDomain);
                result(i,j) = multByElementSum(tmp, kernel);
            }
        }

        return result;

    }
    else{
        if(!strcmp(type, "full")){

        }
    }


}

template <class T>
Array<T, 3> convolve(const Array<T, 3>& x, const Array<T, 3>& kernel, char* type)
{

}


#endif // UTIL_H
