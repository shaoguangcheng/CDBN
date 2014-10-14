/*!
 * \file matrixOperation.h
 * \breif Some useful matrix operations are implemented,
 *      including matrix multiplication, division, addition
 *      and convlution. For some operations, both 2d and 3d
 *      versions are provided. All these functions have been tested.
 *
 * \author Shaoguang Cheng. From Xi'an, China
 * \date   2014.9.29
 */

#ifndef MATRIXOPERATION_H
#define MATRIXOPERATION_H

#include <blitz/array.h>
#include <blitz/funcs.h>
#include <blitz/vector-et.h>
#include <random/normal.h>

#include <string.h>
#include <time.h>

#include "global.h"

#ifdef BZ_NAMESPACES
using namespace blitz;
#endif

using namespace ranlib;

#ifndef COMMENT
#define COMMENT 0
#endif

/*!
 *  \brief transpose compute the transpose of \a x for 2d matrix
 *  \param \a x 2d matrix
 */
template <class T>
Array<T, 2> transpose(Array<T, 2> x)
{
    return x.transpose(secondDim, firstDim);
}

/*!
 *  \brief transpose compute the transpose of \a x for 3d matrix
 *  \param \a x 3d matrix
 */
template <class T>
Array<T,3> transpose(Array<T, 3> x)
{
    return x.transpose(thirdDim, secondDim, firstDim);
}

//////////////////////// multiplication /////////////////////
/*!
 * \brief multByElementSum compute c = sumij(Xij*Yij) for 2d matrix
 * \param \a x matrix
 * \param \a y matrix
 * \note \a x and \a y have the same size
 */
template <class T>
T multByElementSum(const Array<T, 2>& x, const Array<T, 2>& y)
{
    if(x.shape()(0) != y.shape()(0)||x.shape()(1) != y.shape()(1)){
        DEBUGMSG("Size can not match");
        exit(EXIT_FAILURE);
    }

    T result=(T)0;
#if COMMENT
    for(int i=0;i<x.shape()(0);++i){
        for(int j=0;j<x.shape()(1);++j){
            result += x(i,j)*y(i,j);
        }
    }
#else
    Array<T, 2> tmp(x.shape());
    firstIndex i;
    secondIndex j;

    tmp = x(i,j)*y(i,j);
    result = sum(tmp);
#endif

    return result;
}

/*!
 * \brief multByElementSum compute c = sumij(Xij*Yij) for 3d array
 * \param \a x 3d array
 * \param \a y 3d array
 * \note \a x and \a y have the same size
 */
template <class T>
T multByElementSum(const Array<T, 3>& x, const Array<T, 3>& y)
{
    if(x.shape()(0) != y.shape()(0)||
            x.shape()(1) != y.shape()(1)||
            x.shape()(2) != y.shape()(2)){
        DEBUGMSG("Size can not match");
        exit(EXIT_FAILURE);
    }

    T result=(T)0;
    Array<T, 3> tmp(x.shape());

    firstIndex i;
    secondIndex j;
    thirdIndex k;

    tmp = x(i,j,k)*y(i,j,k);
    result = sum(tmp);

    return result;
}

/*!
 * \brief multByElement compute c = (Xij*Yij)
 * \param \a x matrix
 * \param \a y matrix
 * \note \a x and \a y have the same size
 */
template <class T>
T multByElement(const Array<T, 2>& x, const Array<T, 2>& y)
{
    if(x.shape()(0) != y.shape()(0)||x.shape()(1) != y.shape()(1)){
        DEBUGMSG("Size can not match");
        exit(EXIT_FAILURE);
    }

    Array<T, 2> tmp(x.shape());
    firstIndex i;
    secondIndex j;

    tmp = x(i,j)*y(i,j);

    return tmp;
}

/*!
 * \brief multScalar compute c = x*y
 * \param \a x vector
 * \param \a y a number
 */
template <class T>
Array<T, 1> multScalar(const Array<T, 1>& x, T y)
{
    Array<T, 1> result(x.shape());
    firstIndex i;

    result = x(i)*y;

    return result;
}

/*!
 * \brief multScalar compute c = x*y
 * \param \a x matrix
 * \param \a y a number
 */
template <class T>
Array<T, 2> multScalar(const Array<T, 2>& x, T y)
{
    Array<T, 2> result(x.shape());
    firstIndex i;
    secondIndex j;

    result = x(i, j)*y;

    return result;
}

/*!
 * \brief multScalar compute c = x*y
 * \param \a x matrix
 * \param \a y a number
 */
template <class T>
Array<T, 3> multScalar(const Array<T, 3>& x, T y)
{
    Array<T, 3> result(x.shape());
    firstIndex i;
    secondIndex j;
    thirdIndex k;

    result = x(i, j, k)*y;

    return result;
}

/*!
 * \brief multScalar compute c = x*y
 * \param \a x matrix
 * \param \a y a number
 */
template <class T>
Array<T, 4> multScalar(const Array<T, 4>& x, T y)
{
    Array<T, 4> result(x.shape());
    firstIndex i;
    secondIndex j;
    thirdIndex k;
    fourthIndex l;

    result = x(i, j, k, l)*y;

    return result;
}

/*!
 * \brief multVector compute c = x*y
 * \param \a x matrix
 * \param \a y a vector
 * \note the cols of \a x must have the same size with the length of \a y
 */
template <class T>
Array<T, 1> multVector(const Array<T, 2>& x, const Array<T, 1>& y)
{
    if(x.cols() != y.shape()(0)){
        DEBUGMSG("Size can not match");
        exit(EXIT_FAILURE);
    }

    firstIndex i;
    secondIndex j;

    Array<T, 1> result(x.shape()(0));
    result = sum(x(i,j)*y(j), j);

    return result;
}

/*!
 * \brief multMatrix compute c = x*y
 * \param \a x matrix
 * \param \a y matrix
 * \note the cols of \a x must have the same size with the rows of \a y
 */
template <class T>
Array<T, 2> multMatrix(const Array<T, 2>& x, const Array<T, 2>& y)
{
    if(x.cols() != y.rows()){
        DEBUGMSG("Size can not match");
        exit(EXIT_FAILURE);
    }

    firstIndex i;
    secondIndex j;
    thirdIndex k;

    Array<T, 2> result(x.rows(), y.cols());
    result = sum(x(i,k)*y(k,j), k);

    return result;
}

///////////////////// division /////////////////////////
/*!
 * \brief divideScalar for 2d operation compute c = x/y
 * \param \a x matrix
 * \param \a y a number
 * \note  \a y must not be zero
 */
template <class T>
Array<T, 2> divideScalar(const Array<T, 2>& x, T y)
{
    if(std::abs(double(y)) < 1e-6){
        DEBUGMSG("Warning : divide by a very little number");
    }

    if(std::abs(double(y)) < 1e-9){
        DEBUGMSG("Can not divide by zero");
        exit(EXIT_FAILURE);
    }

    firstIndex i;
    secondIndex j;
    Array<T, 2> result(x.shape());

    result = x(i,j)/(double)(y);

    return result;
}

/*!
 * \brief divideScalar for 3d operation compute c = x/y
 * \param \a x 3d array
 * \param \a y a number
 * \note  \a y must not be zero
 */
template <class T>
Array<T, 3> divideScalar(const Array<T, 3>& x, T y)
{
    if(std::abs(double(y)) < 1e-6){
        DEBUGMSG("Warning : divide by a very little number");
    }

    if(std::abs(double(y)) < 1e-9){
        DEBUGMSG("Can not divide by zero");
        exit(EXIT_FAILURE);
    }

    firstIndex i;
    secondIndex j;
    thirdIndex k;
    Array<T, 3> result(x.shape());

    result = x(i,j,k)/(double)(y);

    return result;
}

/*!
 * \brief divideByScalar for 2d operation compute c = x/y
 * \param \a x matrix
 * \param \a y a number
 * \note  \a y must not be zero
 */
template <class T>
Array<T, 2> divideByScalar(const Array<T, 2>& x, T y)
{
    if(blitz::any(blitz::abs(x) <= 1e-6)){
        DEBUGMSG("Warning : divide by a very little number");
    }

    if(blitz::any(blitz::abs(x) <= 1e-9)){
        DEBUGMSG("Can not divide by zero");
        exit(EXIT_FAILURE);
    }

    firstIndex i;
    secondIndex j;
    Array<T, 2> result(x.shape());

    result = (double)(y)/x(i,j);

    return result;
}

/*!
 * \brief divideByScalar for 3d operation compute c = x/y
 * \param \a x 3d array
 * \param \a y a number
 * \note  \a y must not be zero
 */
template <class T>
Array<T, 3> divideByScalar(const Array<T, 3>& x, T y)
{
    if(blitz::any(blitz::abs(x) <= 1e-6)){
        DEBUGMSG("Warning : divide by a very little number");
    }

    if(blitz::any(blitz::abs(x) <= 1e-9)){
        DEBUGMSG("Can not divide by zero");
        exit(EXIT_FAILURE);
    }

    firstIndex i;
    secondIndex j;
    thirdIndex k;
    Array<T, 3> result(x.shape());

    result = (double)(y)/x(i,j,k);

    return result;
}

/*!
 * \brief divideByScalar for 4d operation compute c = x/y
 * \param \a x 4d array
 * \param \a y a number
 * \note  \a y must not be zero
 */
template <class T>
Array<T, 4> divideByScalar(const Array<T, 4>& x, T y)
{
    if(blitz::any(blitz::abs(x) <= 1e-6)){
        DEBUGMSG("Warning : divide by a very little number");
    }

    if(blitz::any(blitz::abs(x) <= 1e-9)){
        DEBUGMSG("Can not divide by zero");
        exit(EXIT_FAILURE);
    }

    firstIndex i;
    secondIndex j;
    thirdIndex k;
    fourthIndex l;

    Array<T, 4> result(x.shape());

    result = (double)(y)/x(i,j,k,l);

    return result;
}

///////////////////// addition /////////////////////////
/*!
 * \brief addScalar for 2d operation compute compute c = Xij+Y
 * \param \a x matrix
 * \param \a y a number
 */
template <class T>
Array<T, 2> addScalar(const Array<T, 2>& x, T y)
{
    firstIndex i;
    secondIndex j;

    Array<T, 2> result(x.shape());
    result = x(i,j)+y;

    return result;
}

/*!
 * \brief addScalar for 3d operation compute compute c = Xij+Y
 * \param \a x 3d array
 * \param \a y a number
 */
template <class T>
Array<T, 3> addScalar(const Array<T, 3>& x, T y)
{
    firstIndex i;
    secondIndex j;
    thirdIndex k;

    Array<T, 3> result(x.shape());
    result = x(i,j,k)+y;

    return result;
}

/*!
 * \brief addScalar for 4d operation compute compute c = Xij+Y
 * \param \a x 4d array
 * \param \a y a number
 */
template <class T>
Array<T, 4> addScalar(const Array<T, 4>& x, T y)
{
    firstIndex i;
    secondIndex j;
    thirdIndex k;
    fourthIndex l;

    Array<T, 4> result(x.shape());
    result = x(i,j,k,l)+y;

    return result;
}

/*!
 * \brief addVectorByCol compute c = x + [y]
 * \param \a x matrix
 * \param \a y a vector
 * \note the cols of x must have the same size with the length of y
 */
template<class T>
Array<T, 2> addVectorByRow(const Array<T, 2>& x, const Array<T, 1>& y)
{
    if(x.cols() != y.shape()[0]){
        DEBUGMSG("Size can not match");
        exit(EXIT_FAILURE);
    }

    firstIndex i;
    secondIndex j;

    Array<T, 2> result(x.shape());
    result = x(i,j) + y(j);

    return result;
}

/*!
 * \brief compute c = x + [y]
 * \param \a x matrix
 * \param \a y a vector
 * \note the rows of x must have the same size with the length of y
 */
template<class T>
Array<T, 2> addVectorByCol(const Array<T, 2>& x, const Array<T, 1>& y)
{
    if(x.rows() != y.shape()[0]){
        DEBUGMSG("Size can not match");
        exit(EXIT_FAILURE);
    }

    firstIndex i;
    secondIndex j;

    Array<T, 2> result(x.shape());
    result = x(i,j) + y(i);

    return result;
}


//////////////////////////// convolution ////////////////////////////
/*!
 * @brief convolve2d do convolution operation for 2d matrix \a x using kernel \a y
 *       \a type can be \a "valid" or \a "full". if type is "valid", edge does not be processed. Otherwise.
 * @param \a x 2d matrix
 * @param \a kernel
 * @param \a type convolution type
 * @return
 */
template <class T>
Array<T, 2> convolve(const Array<T, 2>& x, Array<T, 2> kernel, const char* type)
{
    TinyVector<int, 2> shapeX = x.shape(), shapeKernel = kernel.shape();

    const int first = 0;
    const int second = 1;

    int n = 0;
    bool isBreak = false;
    for(int i = 0; i < shapeKernel[first]; ++i){
        for(int j = 0;j < shapeKernel[second]; ++j){

            n++;
            if(2*n > kernel.size()){
                isBreak = true;
                break;
            }

            T tmp = kernel(i, j);
            kernel(i,j) = kernel(shapeKernel[first]-i-1, shapeKernel[second]-j-1);
            kernel(shapeKernel[first]-i-1, shapeKernel[second]-j-1) = tmp;
        }

        if(isBreak){
            break;
        }

    }

    if(!strcmp(type, "valid")){
        if(shapeKernel(first) > shapeX(first)||
                shapeKernel(second) > shapeX(second)){
            DEBUGMSG("In valid mode, kernel size can not be larger than data size");
            exit(EXIT_FAILURE);
        }

        Array<T, 2> result(shapeX(first)-shapeKernel(first)+1,
                           shapeX(second)-shapeKernel(second)+1);
        for(int i = 0; i < result.shape()[first]; ++i){
            for(int j = 0;j < result.shape()[second]; ++j){
                TinyVector<int, 2> lowerBound(i,j),
                        upperBound(i+shapeKernel(first)-1, j+shapeKernel(second)-1);
                RectDomain<2> subDomain(lowerBound, upperBound);
                Array<T, 2> tmp = x(subDomain);
                result(i,j) = multByElementSum(tmp, kernel);
            }
        }

        return result;

    }
    else{
        if(!strcmp(type, "full")){
            Array<T, 2> result(shapeX(first)+shapeKernel(first)-1,
                               shapeX(second)+shapeKernel(second)-1);
            Array<T, 2> fullX(shapeX(first)+2*(shapeKernel(first)-1),
                              shapeX(second)+2*(shapeKernel(second)-1));
            fullX = 0;
            TinyVector<int, 2> lBound(shapeKernel(first)-1,
                                      shapeKernel(second)-1),
                    uBound(shapeKernel(first)+shapeX(first)-2,
                           shapeKernel(second)+shapeX(second)-2);
            RectDomain<2>  domain(lBound, uBound);
            fullX(domain) = x;
            for(int i = 0; i < result.shape()[first]; ++i){
                for(int j = 0; j <result.shape()[second]; ++j){
                    TinyVector<int, 2> lowerBound(i,j),
                            upperBound(i+shapeKernel(first)-1, j+shapeKernel(second)-1);
                    RectDomain<2> subDomain(lowerBound, upperBound);
                    Array<T, 2> tmp = fullX(subDomain);
                    result(i,j) = multByElementSum(tmp, kernel);
                }
            }

            return result;
        }
        else{
            DEBUGMSG("Undefined convolution type");
            exit(EXIT_FAILURE);
        }
    }
}

/*!
 * @brief convolve3d do convolution operation for 3d matrix \a x using kernel \a y
 *       \a type can be \a "valid" or \a "full". if type is "valid", edge does not be processed. Otherwise.
 * @param \a x 3d matrix
 * @param \a kernel
 * @param \a type convolution type
 * @return
 */
template <class T>
Array<T, 3> convolve(const Array<T, 3>& x, Array<T, 3> kernel, const char* type)
{
    TinyVector<int, 3> shapeX = x.shape(), shapeKernel = kernel.shape();

    const int first  = 0;
    const int second = 1;
    const int third  = 2;

    int n = 0;
    bool isBreak = false;
    for(int i = 0; i < shapeKernel[first]; ++i){
        for(int j = 0;j < shapeKernel[second]; ++j){
            for(int k = 0; k < shapeKernel[third]; ++k){

                n++;
                if(2*n > kernel.size()){
                    isBreak = true;
                    break;
                }

                T tmp = kernel(i, j, k);
                kernel(i,j) = kernel(shapeKernel[first]-i-1, shapeKernel[second]-j-1, shapeKernel[third]-k-1);
                kernel(shapeKernel[first]-i-1, shapeKernel[second]-j-1, shapeKernel[third]-k-1) = tmp;
            }

            if(isBreak){
                break;
            }

        }

        if(isBreak){
            break;
        }

    }



    if(!strcmp(type, "valid")){
        if(shapeKernel(first) > shapeX(first)||
                shapeKernel(second) > shapeX(second)||
                shapeKernel(third) > shapeX(third)){
            DEBUGMSG("In valid mode, kernel size can not be larger than data size");
            exit(EXIT_FAILURE);
        }

        Array<T, 3> result(shapeX(first)-shapeKernel(first)+1,
                           shapeX(second)-shapeKernel(second)+1,
                           shapeX(third)-shapeKernel(third)+1);

        for(int i = 0; i < result.shape()[first]; ++i){
            for(int j = 0; j < result.shape()[second]; ++j){
                for(int k = 0; k < result.shape()[third]; ++k){
                    TinyVector<int, 3> lowerBound(i, j, k),
                            upperBound(i+shapeKernel(first)-1,
                                       j+shapeKernel(second)-1,
                                       k+shapeKernel(third)-1);
                    RectDomain<3> subDomain(lowerBound, upperBound);
                    Array<T, 3> tmp = x(subDomain);

                    result(i,j,k) = multByElementSum(tmp, kernel);
                }
            }
        }

        return result;
    }
    else{
        if(!strcmp(type, "full")){
            Array<T, 3> result(shapeX(first)+shapeKernel(first)-1,
                               shapeX(second)+shapeKernel(second)-1,
                               shapeX(third)+shapeKernel(third)-1);
            Array<T, 3> fullX(shapeX(first)+2*(shapeKernel(first)-1),
                              shapeX(second)+2*(shapeKernel(second)-1),
                              shapeX(third)+2*(shapeKernel(third)-1));
            fullX = 0;
            TinyVector<int, 3> lBound(shapeKernel(first)-1,
                                      shapeKernel(second)-1,
                                      shapeKernel(third)-1),
                    uBound(shapeKernel(first)+shapeX(first)-2,
                           shapeKernel(second)+shapeX(second)-2,
                           shapeKernel(third)+shapeX(third)-2);
            RectDomain<3>  domain(lBound, uBound);
            fullX(domain) = x;

            for(int i = 0; i < result.shape()[first]; ++i){
                for(int j = 0; j < result.shape()[second]; ++j){
                    for(int k = 0; k < result.shape()[third]; ++k){
                        TinyVector<int, 3> lowerBound(i, j, k),
                                upperBound(i+shapeKernel(first)-1,
                                           j+shapeKernel(second)-1,
                                           k+shapeKernel(third)-1);
                        RectDomain<3> subDomain(lowerBound, upperBound);
                        Array<T, 3> tmp = x(subDomain);
                        result(i,j,k) = multByElementSum(tmp, kernel);
                    }
                }
            }

            return result;
        }
        else{
            DEBUGMSG("Undefined convolution type");
            exit(EXIT_FAILURE);
        }
    }
}

/*!
 * @brief convolve do convolution operation for a set of 2d matrix \a x using kernel \a y
 *       \a type can be \a "valid" or \a "full". if type is "valid", edge does not be processed. Otherwise.
 * @param \a x a set of 2d matrix
 * @param \a kernel
 * @param \a type convolution type
 * @return
 */
template <class T>
Array<T, 3> convolve(const Array<T, 3>& x, const Array<T, 2>& kernel, const char* type)
{
    TinyVector<int, 3> shapeX = x.shape();
    TinyVector<int, 2> shapeKernel = kernel.shape();
    TinyVector<int, 3> shapeResult;

    Array<T, 3> result;

    int nCase = shapeX(2); /// number of cases

    shapeResult(2) = shapeX(2);

    if(!strcmp(type, "valid")){
        shapeResult(0) = shapeX(0) - shapeKernel(0) + 1;
        shapeResult(1) = shapeX(1) - shapeKernel(1) + 1;
        result.resize(shapeResult);
    }
    else{
        if(!strcmp(type, "full")){
            shapeResult(0) = shapeX(0) + shapeKernel(0) - 1;
            shapeResult(1) = shapeX(1) + shapeKernel(1) - 1;
            result.resize(shapeResult);
        }
        else{
            DEBUGMSG("Undefined convolution type");
            exit(EXIT_FAILURE);
        }
    }

    for(int i = 0; i < nCase; ++i){
        result(Range::all(), Range::all(), i) = convolve(x(Range::all(), Range::all(), i), kernel, type);
    }

    return result;
}

/*!
 * @brief convolve do convolution operation for a set of 3d array \a x using kernel \a y
 *       \a type can be \a "valid" or \a "full". if type is "valid", edge does not be processed. Otherwise.
 * @param \a x a set of 3d array
 * @param \a kernel
 * @param \a type convolution type
 * @return
 */
template <class T>
Array<T, 4> convolve(const Array<T, 4>& x, const Array<T, 3>& kernel, const char* type)
{
    TinyVector<int, 4> shapeX = x.shape();
    TinyVector<int, 3> shapeKernel = kernel.shape();
    TinyVector<int, 4> shapeResult;

    Array<T, 4> result;

    int nCase = shapeX(3); /// number of cases

    shapeResult(3) = shapeX(3);

    if(!strcmp(type, "valid")){
        shapeResult(0) = shapeX(0) - shapeKernel(0) + 1;
        shapeResult(1) = shapeX(1) - shapeKernel(1) + 1;
        shapeResult(2) = shapeX(2) - shapeKernel(2) + 1;
        result.resize(shapeResult);
    }
    else{
        if(!strcmp(type, "full")){
            shapeResult(0) = shapeX(0) + shapeKernel(0) - 1;
            shapeResult(1) = shapeX(1) + shapeKernel(1) - 1;
            shapeResult(2) = shapeX(2) + shapeKernel(2) - 1;
            result.resize(shapeResult);
        }
        else{
            DEBUGMSG("Undefined convolution type");
            exit(EXIT_FAILURE);
        }
    }

    for(int i = 0; i < nCase; ++i){
        result(Range::all(), Range::all(), Range::all(), i) = convolve(x(Range::all(), Range::all(), Range::all(), i), kernel, type);
    }

    return result;
}

#if 0
template <class T, int DIM>
Array<T, DIM+1> convolve(const Array<T, DIM+1>& x, const Array<T, DIM>& kernel, char* type)
{
    TinyVector<int, DIM+1> shapeX = x.shape();
    TinyVector<int, DIM> shapeKernel = kernel.shape();
    TinyVector<int, DIM+1> shapeResult;

    Array<T, DIM+1> result;

    int nCase = shapeX(DIM); /// number of cases

    shapeResult(DIM) = shapeX(DIM);

    if(!strcmp(type, "valid")){
        for(int i = 0; i < DIM; ++i){
            shapeResult(i) = shapeX(i) - shapeKernel(i) + 1;
        }

        result.resize(shapeResult);
    }
    else{
        if(!strcmp(type, "full")){
            for(int i = 0; i < DIM; ++i){
                shapeResult(i) = shapeX(i) + shapeKernel(i) - 1;
            }

            result.resize(shapeResult);
        }
        else{
            DEBUGMSG("Undefined convolution type");
            exit(EXIT_FAILURE);
        }
    }

    if(2 == DIM){
        for(int i = 0; i < nCase; ++i){
            result(Range::all(), Range::all(), i) = convolve(x(Range::all(), Range::all(), i), kernel, type);
        }
    }
    else{
        if(3 == DIM){
            for(int i = 0; i < nCase; ++i){
                result(Range::all(), Range::all(), Range::all(), i) = convolve(x(Range::all(), Range::all(), Range::all(), i), kernel, type);
            }
        }
        else{
            DEBUGMSG("Undefined operation");
            exit(EXIT_FAILURE);
        }
    }

    return result;
}
#endif

//////////////////////////// special function /////////////////////////
/*!
 * @brief sigmod function f = 1/(1+exp(-x))
 *        Both for 2d and 3d
 * @param x input 2d matrix
 * @return function value
 */
template <class T, int DIM>
Array<T, DIM> sigmod(const Array<T, DIM>& x)
{
    Array<T, DIM> result(x.shape());

    result = multScalar(x, (T)(-1));
    result = blitz::exp(result);
    result = addScalar(result, T(1));
    result = divideByScalar(result, (T)(1));

    return result;
}

/**
 * @brief randn generator normal distribution for 2d matrix
 * @param shape the size of 2d matrix
 * @return
 */
template <class T>
Array<T, 2> randn(Array<T, 2> x)
{
    Normal<T> rng(0, 1);
    rng.seed((unsigned int)time(0));

    TinyVector<int, 2> shape(x.shape());
    for(int i = 0; i < shape(0); ++i){
        for(int j = 0; j < shape(1); ++j){
                x(i,j) = rng.random();
        }
    }

    return x;
}

/**
 * @brief randn generator normal distribution for 3d array
 * @param shape the size of 3d array
 * @return
 */
template <class T>
Array<T, 3> randn(Array<T, 3> x)
{
    Normal<T> rng(0, 1);
    rng.seed((unsigned int)time(0));

    TinyVector<int, 3> shape(x.shape());
    for(int i = 0; i < shape(0); ++i){
        for(int j = 0; j < shape(1); ++j){
            for(int k = 0; k < shape(2); ++k){
                x(i,j,k) = rng.random();
            }
        }
    }

    return x;
}

/**
 * @brief randn generator normal distribution for 4d array
 * @param shape the size of 4d array
 * @return
 */
template <class T>
Array<T, 4> randn(Array<T, 4> x)
{
    Normal<T> rng(0, 1);
    rng.seed((unsigned int)time(0));

    TinyVector<int, 4> shape(x.shape());
    for(int i = 0; i < shape(0); ++i){
        for(int j = 0; j < shape(1); ++j){
            for(int k = 0; k < shape(2); ++k){
                for(int l = 0; l < shape(3); ++l){
                    x(i,j,k) = rng.random();
                }
            }
        }
    }

    return x;
}

template <class T>
void square(Array<T, 4>& x)
{
    TinyVector<int, 4> shapeX = x.shape();

    for(int i = 0; i < shapeX(0); ++i){
        for(int j = 0; j < shapeX(1); ++j){
            for(int k = 0; k < shapeX(2); ++k){
                for(int l = 0; l < shapeX(3); ++l){
                    x(i, j, k, l) *= x(i, j, k, l);
                }
            }
        }
    }
}

template <class T>
void square(Array<T, 5>& x)
{
    TinyVector<int, 5> shapeX = x.shape();

    for(int i = 0; i < shapeX(0); ++i){
        for(int j = 0; j < shapeX(1); ++j){
            for(int k = 0; k < shapeX(2); ++k){
                for(int l = 0; l < shapeX(3); ++l){
                    for(int r = 0; r < shapeX(4); ++r){
                        x(i, j, k, l, r) *= x(i, j, k, l, r);
                    }
                }
            }
        }
    }
}

template <class T>
T euclidNorm(Array<T, 4>& x)
{
    square(x);

    return sum(x);
}

template <class T>
T euclidNorm(Array<T, 5>& x)
{
    square(x);

    return sum(x);
}

#endif // MATRIXOPERATION_H
