/*!
 * \file util.h
 * \breif Some useful functions are implemented,
 *      including shuffle, pooling and ceiling. For some operations, both 2d and 3d
 *      versions are provided. All these functions have been tested.
 *
 * \author Shaoguang Cheng. From Xi'an, China
 * \date   2014.10.9
 */

#ifndef UTIL_H
#define UTIL_H

#include <blitz/array.h>
#include <blitz/vector-et.h>
#include <random/uniform.h>

#include <algorithm>
#include <time.h>

#include "global.h"

#ifdef BZ_NAMESPACES
using namespace blitz;
#endif

using namespace ranlib;

using namespace std;

/*!
 * @brief shuffle data for 2d image
 */
template <class T>
Array<T, 4> shuffleData(const Array<T, 4>& x)
{
    int nCase = x.shape()(2);
    int *array = new int [nCase];

    for(int i=0;i<nCase;i++){
        array[i] = i;
    }

    random_shuffle(array, array+nCase);

    Array<T, 4> result(x.shape());
    for(int i=0;i<nCase;++i){
        result(Range::all(), Range::all(), i, Range::all()) = x(Range::all(), Range::all(), array[i], Range::all());
    }

    delete [] array;

    return result;
}

/*!
 * @brief shuffle data for 3d shape
 */
template <class T>
Array<T, 5> shuffleData(const Array<T, 5>& x)
{
    int nCase = x.shape()(3);
    int *array = new int [nCase];

    for(int i=0;i<nCase;i++){
        array[i] = i;
    }

    random_shuffle(array, array+nCase);

    Array<T, 5> result(x.shape());
    for(int i=0;i<nCase;++i){
        result(Range::all(), Range::all(), Range::all(), i, Range::all()) = x(Range::all(), Range::all(), Range::all(), array[i], Range::all());
    }

    delete [] array;

    return result;
}

template <class T>
void maxPooling(Array<T, 2>& P, Array<T, 2>& state, Array<T, 2>& outPooling, int scale)
{
    int rowOfOut = P.rows()/scale, colOfOut = P.cols()/scale;

    Uniform<T> rng;
    rng.seed((unsigned int)time(0));

    T uniformNumber;

    int p = 0, q = 0;
    for(int i = 0; i < rowOfOut; ++i){
        for(int j = 0; j < colOfOut; ++j){

            //find the index of maximum P in scale*scale area
            int indexM = 0, indexN = 0;
            for(int m = 0; m < scale; ++m){
                for(int n = 0; n < scale; ++n){
                    if(P(i*scale+indexM, j*scale+indexN) < P(i*scale+m, j*scale+n)){
                        indexM = m;
                        indexN = n;
                    }
                }
            }

            uniformNumber = rng.random();

            if(uniformNumber < P(i*scale+indexM, j*scale+indexN)){
                //set the state of convolutional layer
                for(int m = 0; m < scale; ++m){
                    for(int n = 0; n < scale; ++n){
                        if(m == indexM && n == indexN){
                            state(i*scale+m,j*scale+n) = T(1);
                        }
                        else{
                            state(i*scale+m,j*scale+n) = T(0);
                        }
                    }
                }
            }
            else{
                for(int m = 0; m < scale; ++m){
                    for(int n = 0; n < scale; ++n){
                        state(i*scale+m,j*scale+n) = T(0);
                    }
                }
            }

            // update the output of pooling layer
            outPooling(p, q) = P(i*scale+indexM, j*scale+indexN);
            ++q;
            if(q == colOfOut){
                q = 0;
            }
        }
        ++p;
    }
}

template <class T>
void maxPooling(Array<T, 3> &P, Array<T, 3> &state, Array<T, 3> &outPooling, int scale)
{
    int rowOfOut = P.rows()/scale, colOfOut = P.cols()/scale, depthOfOut = P.depth()/scale;

    Uniform<T> rng;
    rng.seed((unsigned int)time(0));

    T uniformNumber;

    int p = 0, q = 0, r = 0;
    for(int i = 0; i < rowOfOut; ++i){
        for(int j = 0; j < colOfOut; ++j){
            for(int k = 0; k < depthOfOut; ++k){

                int indexM = 0, indexN = 0, indexL = 0;
                for(int m = 0; m < scale; ++m){
                    for(int n = 0; n < scale; ++n){
                        for(int l = 0; l < scale; ++l){
                            if(P(i*scale+indexM, j*scale+indexN, k*scale+indexL) < P(i*scale+m, j*scale+n, k*scale+l)){
                                indexM = m;
                                indexN = n;
                                indexL = l;
                            }
                        }
                    }
                }

                uniformNumber = rng.random();

                if(uniformNumber < P(i*scale+indexM, j*scale+indexN, k*scale+indexL)){
                    for(int m = 0; m < scale; ++m){
                        for(int n = 0; n < scale; ++n){
                            for(int l = 0; l < scale; ++l){
                                if(m == indexM && n == indexN && l == indexL){
                                    state(i*scale+m,j*scale+n,k*scale+l) = T(1);
                                }
                                else{
                                    state(i*scale+m,j*scale+n,k*scale+l) = T(0);
                                }
                            }
                        }
                    }
                }
                else{
                    for(int m = 0; m < scale; ++m){
                        for(int n = 0; n < scale; ++n){
                            for(int l = 0; l < scale; ++l){
                                state(i*scale+m,j*scale+n,k*scale+l) = T(0);
                            }
                        }
                    }
                }

                outPooling(p,q,r) = P(i*scale+indexM,j*scale+indexN,k*scale+indexL);
                ++r;
                if(r == depthOfOut){
                    r = 0;
                }
            }
            ++q;
            if(q == colOfOut){
                q = 0;
            }
        }
        ++p;
        if(p == rowOfOut){
            p = 0;
        }
    }
}

/**
 * @brief stochasticPooling implement stochastic pooling algorithm, still some questions to be fixed
 * @param P original data to pooling
 * @param state one of output
 * @param outPooling the pooling result
 * @param scale size of block to pooling
 */
template <class T>
void stochasticPooling(Array<T, 2>& P, Array<T, 2>& state, Array<T, 2>& outPooling, int scale)
{
    if(P.shape()(0) != outPooling.shape()(0)*scale || P.shape()(1) != outPooling.shape()(1)*scale){
        DEBUGMSG("output of pooling operation occurs error");
        exit(EXIT_FAILURE);
    }

    int rowOfOut = P.rows()/scale, colOfOut = P.cols()/scale;

    Uniform<T> rng;
    rng.seed((unsigned int)time(0));

    T uniformNumber;

    int p = 0, q = 0;
    for(int i = 0; i < rowOfOut; ++i){
        for(int j = 0; j < colOfOut; ++j){

            T sum = T(0);
            for(int m = 0; m < scale; ++m){
                for(int n = 0; n < scale; ++n){
                    sum += P(i*scale+m, j*scale+n);
                }
            }

            sum += T(1); // this line is useful in CRBM, one way to make the hidden layer more sparse.
                      // However, in CNN we do not need this operation

            int k = 0;
            Array<T ,1> tmp(scale*scale+1);
            tmp = 0;
            T sumTmp = T(0);
            for(int m = 0; m < scale; ++m){
                for(int n = 0; n < scale; ++n){
                    sumTmp += P(i*scale+m, j*scale+n)/sum;
                    tmp(k++) = sumTmp;

                    P(i*scale+m, j*scale+n) /= sum; // Note : here must be careful
                }
            }

            tmp(k) = T(1);
            uniformNumber = rng.random();

            int index = -1;
            for(int m = 0; m < tmp.size(); ++m){
                if(uniformNumber < tmp(m)){
                    index = m;
                    break;
                }
            }

            for(int m = 0; m < scale; ++m){
                for(int n = 0; n < scale; ++n){
                    if(m*scale+n == index){
                        state(i*scale+m, j*scale+n) = T(1);
                    }
                    else{
                        state(i*scale+m, j*scale+n) = T(0);
                    }
                }
            }

            if(index == k){
                outPooling(p,q) = 0;
            }
            else{
                outPooling(p,q) = P(i*scale+index/scale, j*scale+index%scale);
            }
            ++q;
            if(q == colOfOut){
                q = 0;
            }
        }
        ++p;
    }
}

/**
 * @brief stochasticPooling implement stochastic pooling algorithm, still some questions to be fixed
 * @param P original data to pooling
 * @param state one of output
 * @param outPooling the pooling result
 * @param scale size of block to pooling
 */
template <class T>
void stochasticPooling(Array<T, 3> P, Array<T, 3>& state, Array<T, 3>& outPooling, int scale)
{
    if(P.rows() != outPooling.rows()*scale ||
            P.cols() != outPooling.cols()*scale ||
            P.depth() != outPooling.depth()*scale){
        DEBUGMSG("output of pooling operation occurs error");
        exit(EXIT_FAILURE);
    }

    int rowOfOut = P.rows()/scale, colOfOut = P.cols()/scale, depthOfOut = P.depth()/scale;

    Uniform<T> rng;
    rng.seed((unsigned int)time(0));

    T uniformNumber;

    int p = 0, q = 0, r = 0;
    for(int i = 0; i < rowOfOut; ++i){
        for(int j = 0; j < colOfOut; ++j){
            for(int k = 0; k < depthOfOut; ++k){

                T sum = T(0);
                for(int m = 0; m < scale; ++m){
                    for(int n = 0; n < scale; ++n){
                        for(int l = 0; l < scale; ++l){
                            sum += P(i*scale+m, j*scale+n, k*scale+l);
                        }
                    }
                }

                sum += T(1);

                int s = 0;
                Array<T, 1> tmp(scale*scale*scale+1);
                tmp = T(0);
                T tmpSum = T(0);
                for(int m = 0; m < scale; ++m){
                    for(int n = 0; n < scale; ++n){
                        for(int l = 0; l < scale; ++l){
                            tmpSum += P(i*scale+m, j*scale+n, k*scale+l)/sum;
                            tmp(s++) = tmpSum;

                            P(i*scale+m, j*scale+n, k*scale+l) /= sum;
                        }
                    }
                }
                tmp(s) = T(1);

                uniformNumber = rng.random();

                int index = -1;
                for(int m = 0; m < tmp.size(); ++m){
                    if(uniformNumber < tmp(m)){
                        index = m;
                        break;
                    }
                }

                for(int m = 0; m < scale; ++m){
                    for(int n = 0; n < scale; ++n){
                        for(int l = 0; l < scale; ++l){
                            if(m*scale*scale+n*scale+l == index){
                                state(i*scale+m, j*scale+n, k*scale+l) = T(1);
                            }
                            else{
                                state(i*scale+m, j*scale+n, k*scale+l) = T(0);
                            }
                        }
                    }
                }

                if(index == s){
                    outPooling(p, q, r) = T(0);
                }
                else{
                    int m = index/(scale*scale);
                    int n = (index - m*scale*scale)/scale;
                    int l = (index - m*scale*scale)%scale;
                    outPooling(p, q, r) = P(i*scale+m, j*scale+n, k*scale+l);
                }

                ++r;
                if(r == depthOfOut){
                    r = 0;
                }

            }
            ++q;
            if(q == colOfOut){
                q = 0;
            }
        }
        ++p;
    }
}


/*!
 * \brief ceil get the least integer number that larger than x/y
 * \param x
 * \param y
 * \return
 */
int ceiling(int x, int y);

#endif // UTIL_H
