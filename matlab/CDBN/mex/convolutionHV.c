
#include <mex.h>
#include <malloc.h>
#include <string.h>
#include <stdlib.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	if(nlhs > 1){
		mexErrMsgTxt("Too many outputs");
	}

	if(nrhs != 3){
		mexErrMsgTxt("Too few arguments");
	}

	double* XPtr, *kernelPtr, *resultPtr;
	int nCase, nDimX, nDimKernel, nFeatureMapVis, nFeatureMapHid, row, col, sizeR, sizeC, sizeSquare, i, j, k, l, numX, numTmpX, numTmpResult, numResult, r, c;
	const int* dimX, *dimKernel;

	XPtr = mxGetPr(prhs[0]);
	nDimX = mxGetNumberOfDimensions(prhs[0]); 
	dimX = mxGetDimensions(prhs[0]);

	row = dimX[0];
	col = dimX[1];
	if(nDimX > 2)
		nCase = dimX[2];
	else
		nCase = 1;

	if(nDimX > 3)
		nFeatureMapHid = dimX[3];
	else
		nFeatureMapHid = 1;

	kernelPtr = mxGetPr(prhs[1]);
	nDimKernel = mxGetNumberOfDimensions(prhs[1]); 
	dimKernel = mxGetDimensions(prhs[1]);
	sizeR = dimKernel[0];
	sizeC = dimKernel[1];

	nFeatureMapVis = mxGetScalar(prhs[2]);

	r = row + sizeR - 1;
	c = col + sizeC - 1;

    int dimResult[4] = {r, c, nCase, nFeatureMapVis};

	plhs[0] = mxCreateNumericArray(4, dimResult, mxDOUBLE_CLASS, mxREAL);
	resultPtr = mxGetPr(plhs[0]);

	numX = row * col;
	numTmpX = numX * nCase;
	numResult = r * c;
	numTmpResult = nCase * numResult;
	sizeSquare = sizeR * sizeC;
	double *tmp = (double*)mxMalloc(numResult * sizeof(double));

	for(i = 0; i < nCase; ++i){
		for(j = 0; j < nFeatureMapVis; ++j){

			for(k = 0; k < nFeatureMapHid; ++k){
				convolution2d(XPtr + k * numTmpX + i * numX, row, col, kernelPtr + k * sizeSquare, sizeR, sizeC, "full", tmp, r, c);

				for(l = 0; l < numResult; ++l){
					resultPtr[j * numTmpResult + i * numResult + l] += tmp[l];
				}

			}
		}
	}

	mxFree(tmp);
}

void convolution2d(double* X, int row, int col, double* kernel, int sizeR, int sizeC, const char* type, double* result, int r, int c)
{
    int i, j, k, l;
    int rr = r + sizeR - 1, cc = c + sizeC - 1;

    double tmp = 0.0;
    double* extendX = (double *)malloc(rr * cc * sizeof(double));

    if(!strcmp(type, "valid")){

        for(i = 0; i < c; ++i){
            for(j = 0; j < r; ++j){

                tmp = 0.0;

                for(k = 0; k < sizeC; ++k){
                    for(l = 0; l < sizeR; ++l){
                        tmp += X[(i + k)*row + (j + l)] * kernel[(sizeC - k - 1) * sizeR + (sizeR - l - 1)];
                    }
                }

                result[i * r + j] = tmp;
            }
        }
        goto end;
    }

    if(!strcmp(type, "full")){

    	memset(extendX, 0, rr * cc* sizeof(double));

        for(i = sizeC - 1; i < c; ++i){
        	for(j = sizeR - 1; j < r; ++j){
        		extendX[i * rr + j] = X[(i - sizeC + 1) * row + j - sizeR + 1];
        	}
        }

        for(i = 0; i < c; ++i){
        	for(j = 0; j < r; ++j){

        		tmp = 0.0;

                for(k = 0; k < sizeC; ++k){
                    for(l = 0; l < sizeR; ++l){
                        tmp += extendX[(i + k) * (r + sizeR - 1) + (j + l)] * kernel[(sizeC - k - 1) * sizeR + (sizeR - l - 1)];
                    }
                }

                result[i * r + j] = tmp;    		
        	}
        }

        goto end;
    }

    fprintf(stderr, "Undefined convolution type");
    goto end;

    end :
    free(extendX);

    return;
}
 