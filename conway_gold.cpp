#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

////////////////////////////////////////////////////////////////////////////////
// export C interface
extern "C" 
void computeGoldSeq(  unsigned* gold_world, unsigned* h_world, int width, int height, int iterations);

extern "C" 
unsigned int compare( const unsigned* reference, const unsigned* data, const unsigned int len, const bool verbose);

extern "C"
void printMatrix(unsigned *u, int h, int w);

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set
////////////////////////////////////////////////////////////////////////////////
void
computeGoldSeq( unsigned* gold_world, unsigned* h_world, int width, int height, int iterations)
{
    // copy h_world to gold_world
    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            gold_world[y*height+x] = h_world[y*height+x];
        }
    }
    for (int i = 0; i < iterations; i++) {
        unsigned* tem = (unsigned*) malloc(width*height*sizeof(unsigned));
        for(int y = 0; y < height; y++) {
            for(int x = 0; x < width; x++) {
                int n = 0; // number of neighbors
                for (int y1 = y - 1; y1 <= y + 1; y1++) {
                    for (int x1 = x - 1; x1 <= x + 1; x1++) {
                        if (gold_world[((y1 + height) % height)*height + ((x1 + width) % width)]) {
                            n++;
                        }
                    }
                }
                if (gold_world[y*height+x]) {
                    n--;
                }
                tem[y*height+x] = (n == 3 || (n == 2 && gold_world[y*height+x]));
            }
        }
        for(int y = 0; y < height; y++) {
            for(int x = 0; x < width; x++) {
                gold_world[y*height+x] = tem[y*height+x];
            }
        }
        // print_matrix(gold_world, height, width);

        free(tem);
    }
}    

unsigned int compare( const unsigned* reference, const unsigned* data, const unsigned int len, const bool verbose)
{
    bool result = true;
    for( unsigned int i = 0; i < len; ++i) 
    {
        if( reference[i] != data[i] ) 
        {
            if( verbose)
                printf("Error: data[%d] = %d, reference[%d] = %d\n", i, data[i], i, reference[i]);
            result = false;
        }
    }
    return result;
}

void printMatrix(unsigned *u, int h, int w) 
{
    for(int i = 0; i < h; i++) {
        for(int j = 0; j < w; j++) {
            printf("%d ", u[i*(w) + j]);
        }
        printf("\n");
    }
    printf("\n");
}



