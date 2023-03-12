#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

////////////////////////////////////////////////////////////////////////////////
// export C interface
extern "C" 
void computeGoldSeq(  unsigned* gold_world, unsigned* h_world, int width, int height, int iterations);

extern "C"
void computeLookupTable(unsigned char* lookup_table, uint dimX, uint dimY);

extern "C" 
unsigned int compare( const unsigned* reference, unsigned* data, const unsigned int len, const bool verbose);

extern "C"
void printMatrix(unsigned *u, int h, int w);

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set
////////////////////////////////////////////////////////////////////////////////

#define DO_CPU_COMPUTE 1

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


                    tem[y*width+x] = (n == 3 || (n == 2 && gold_world[y*width+x]));
                }
            }

            for(int y = 0; y < height; y++) {
                for(int x = 0; x < width; x++) {
                    gold_world[y*width+x] = tem[y*width+x];
                }
            }

            free(tem);

            // printf("Iteration %d:\n", i + 1);
            // printMatrix(gold_world, height, width);
        }
    }
}    

// lookup table for 6 x 3 area --> map to 4 bits
// total lookup table size ~ 256kB

inline uint getCellState(uint x, uint y, uint dimX, uint dimY, uint key) {
    uint index = y * dimX + x;
    return (key >> ((dimY * dimX - 1) - index)) & 0x1;
}

void computeLookupTable(unsigned char* lookup_table, uint dimX, uint dimY) {
    uint size = 1 << (dimX * dimY);
    for (uint i = 0; i < size ; i++) {
        for (uint cell = 0; cell < 4; cell++) {
            uint n = 0; // number of neighbors
            for (uint x = 0; x < 3; ++x) {
				for (uint y = 0; y < 3; ++y) {
					n += getCellState(x + cell, y, dimX, dimY, i);
                }
			}
            uint curState = getCellState(1 + cell, 1, dimX, dimY, i);
            n -= curState;

            lookup_table[i] |= (n == 3 || (n == 2 && curState)) << (3 - cell);
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
                // if (counts < 100) {
                //     printf("Error: data[%d] = %d, reference[%d] = %d\n", i, data[i], i, reference[i]);
                // }
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
    int newH = h;
    int newW = w;
    if (h > 16) {
        newH = 16;
    }
    if (w > 40) {
        newW = 40;
    }
    for(int i = 0; i < newH; i++) {
        for(int j = 0; j < newW; j++) {
        // for(int j = w - newW; j < w; j++) {
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



