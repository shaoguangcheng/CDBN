
#include <stdio.h>
#include <malloc.h>

void convolution2d(float* X, int row, int col, float* kernel, int size, const char* type, float* result, int r, int c)
{
    int i, j, k, l;
    int index;

    float tmp = 0.0;

    if(!strcmp(type, "valid")){

        if(r != row - size + 1 || c != col - size + 1){
            printf("Size can not match");
            exit(-1);
        }

        for(i = 0; i < r; ++i){
            for(j = 0; j < c; ++j){

                tmp = 0.0;

                for(k = 0; k < size; ++k){
                    for(l = 0; l < size; ++l){
                        tmp += X[(i + k)*col + (j+l)] * kernel[(size - k - 1) * size + (size - l - 1)];
                        printf("%d\n", (i + k)*c + (j+l));
                    }
                }

                result[i * c + j] = tmp;
            }
        }
    }

    return;
}



int main()
{
	float X[12] = {1,2,3,4,5,6,7,8,9,1,2,3};
	int row = 3, col = 4;

	int i;

	float kernel[4] = {1,2,3,2};
	int size = 2;

	const char* type = "valid";

	float result[6];

	convolution2d(X, row, col, kernel, size, type, result, row - size + 1, col - size + 1);

	for(i = 0; i < (row - size + 1) * (col - size + 1); ++i){
		printf("%f ", result[i]);
	}

	printf("\n");
}