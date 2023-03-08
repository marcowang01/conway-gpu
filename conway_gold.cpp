#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

////////////////////////////////////////////////////////////////////////////////
// export C interface
extern "C" 
void computeGoldSeq(  unsigned* gold_world, unsigned* h_world, int width, int height, int iterations);

extern "C" 
unsigned int compare( const unsigned* reference, unsigned* data, const unsigned int len, const bool verbose);

extern "C"
void printMatrix(unsigned *u, int h, int w);

#define DO_CPU_COMPUTE 0

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set
////////////////////////////////////////////////////////////////////////////////
void
computeGoldSeq( unsigned* gold_world, unsigned* h_world, int width, int height, int iterations)
{
    // copy h_world to gold_world
    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            gold_world[y*width+x] = h_world[y*width+x];
        }
    }
    if (DO_CPU_COMPUTE) {
        for (int i = 0; i < iterations; i++) {
        
            unsigned* tem = (unsigned*) malloc(width*height*sizeof(unsigned));
            for(int y = 0; y < height; y++) {
                for(int x = 0; x < width; x++) {
                    int n = 0; // number of neighbors
                    int x1 = (x-1+width)%width;
                    int x2 = (x+1)%width;
                    int y1 = (y-1+height)%height;
                    int y2 = (y+1)%height;
                    n += gold_world[y1*width+x1] + gold_world[y1*width+x] + gold_world[y1*width+x2] 
                    + gold_world[y*width+x1] + gold_world[y*width+x2] + gold_world[y2*width+x1] 
                    + gold_world[y2*width+x] + gold_world[y2*width+x2];

                    tem[y*height+x] = (n == 3 || (n == 2 && gold_world[y*height+x]));
                }
            }

            for(int y = 0; y < height; y++) {
                for(int x = 0; x < width; x++) {
                    gold_world[y*width+x] = tem[y*width+x];
                }
            }
            // print_matrix(gold_world, height, width);

            free(tem);
        }
    }
   
}    

unsigned int compare( const unsigned* reference, unsigned* data, const unsigned int len, const bool verbose)
{
    bool result = true;
    int counts = 0;
    for( unsigned int i = 0; i < len; ++i) 
    {
        if( reference[i] != data[i] ) 
        {
            if( verbose) {
                printf("Error: data[%d] = %d, reference[%d] = %d\n", i, data[i], i, reference[i]);
                data[i] = 8;
            }
            result = false;
            counts += 1;
        }
    }
    if (counts > 0) {
        printf("Error: %d cells are wrong\n", counts);
    }
    // if (verbose) {
    //     printMatrix(data, 32, 32);
    // }
    return result;
}

void printMatrix(unsigned *u, int h, int w) 
{
    for(int i = 0; i < h; i++) {
        for(int j = 0; j < w; j++) {
            int number = u[i*(w) + j];
            if (number != 1 && number != 0) {
                printf("\033[1;31m%d \033[0m", u[i*(w) + j]); 
            } else {
                // print 1's as green and 0's as white
                if (number == 1) {
                    printf("\033[1;32m%d \033[0m", u[i*(w) + j]); 
                } else {
                    printf("%d ", u[i*(w) + j]); 
                }
            }
            if (j % 8 == 7) {
                printf(" ");
            }
        }
        printf("\n");
        if (i % 8 == 7) {
            printf("\n");
        }
    }
    printf("\n");
}



