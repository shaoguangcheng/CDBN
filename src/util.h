#ifndef UTIL_H
#define UTIL_H

#include <blitz/array.h>

#include <algorithm>

#include "global.h"

#ifdef BZ_NAMESPACES
using namespace blitz;
#endif

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

/*!
 * \brief ceil get the least integer number that larger than x/y
 * \param x
 * \param y
 * \return
 */
int ceiling(int x, int y);

#endif // UTIL_H
