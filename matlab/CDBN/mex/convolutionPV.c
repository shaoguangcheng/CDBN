
#include <mex.h>
#include <malloc.h>
#include <string.h>
#include <stdlib.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	if(nlhs > 1){
		mexErrMsgTxt("Too many outputs");
	}

	if(nrhs != 2){
		mexErrMsgTxt("Too few arguments");
	}

	const char* type = "valid";

	double* dataPtr, *PPtr, *resultPtr;
	int nCase, nDimData, nDimP, nFeatureMapVis, nFeatureMapHid, row, col, sizeR, sizeC, sizeResult, i, j, k, l, numData, numTmpData, numTmpP, numP, r, c;
	const int* dimData, *dimP;

	dataPtr = mxGetPr(prhs[0]);	
	nDimData = mxGetNumberOfDimensions(prhs[0]); 
	dimData = mxGetDimensions(prhs[0]);

	row = dimData[0];
	col = dimData[1];
	if(nDimData > 2)
		nCase = dimData[2];
	else
		nCase = 1;

	if(nDimData > 3)
		nFeatureMapVis = dimData[3];
	else
		nFeatureMapVis = 1;

	PPtr = mxGetPr(prhs[1]);
	nDimP = mxGetNumberOfDimensions(prhs[1]); 
	dimP = mxGetDimensions(prhs[1]);
	sizeR = dimP[0];
	sizeC = dimP[1];

	if(nCase != dimP[2]){
		mexErrMsgTxt("Size can not match");		
	}

	if(nDimP > 3)
		nFeatureMapHid = dimP[3];
	else
		nFeatureMapHid = 1;

	r = row - sizeR + 1;
	c = col - sizeC + 1;

	int dimResult[3] = {r, c, nFeatureMapHid};

	plhs[0] = mxCreateNumericArray(3, dimResult, mxDOUBLE_CLASS, mxREAL);
	resultPtr = mxGetPr(plhs[0]);

	numData = row * col;
	numTmpData = numData * nCase;
	numP = sizeR * sizeC;
	numTmpP = nCase * numP;
	sizeResult = r * c;
	double *tmp = (double*)mxMalloc(sizeResult * sizeof(double));

	mexPrintf("%d %d %d\n", nFeatureMapHid, nFeatureMapVis, nCase);

	for(i = 0; i < nFeatureMapHid; ++i){
		for(k = 0; k < nCase; ++k){
			for(j = 0; j < nFeatureMapVis; ++j){
				convolution2d(dataPtr + j * numTmpData + k * numData, row, col, PPtr + i *numTmpP + k * numP, sizeR, sizeC, type, tmp, r, c);

				for(l = 0; l < sizeResult; ++l){
					resultPtr[i * sizeResult + l] += tmp[l];
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