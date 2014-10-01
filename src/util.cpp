#include "util.h"

int ceiling(int x, int y)
{
    if(0 == y){
        DEBUGMSG("Can not divide by zero");
        exit(EXIT_FAILURE);
    }

    if(0 == x%y){
        return x/y;
    }
    else{
        return x/y+1;
    }
}
