#ifdef _WIN32
#  define NOMINMAX 
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cutil.h>

# define WORLD_WIDTH  5
# define WORLD_HEIGHT 5
# define ITERATIONS   10

////////////////////////////////////////////////////////////////////////////////
// main test routine
void runTest( int argc, char** argv );

void randomInit( unsigned int* world );
void customInit(unsigned int* world, int (*coords)[2], int len);

extern "C" 
void computeGoldSeq(  unsigned* reference, unsigned* idata, int width, int height, int iterations);

extern "C"
void print_matrix(unsigned *u, int h, int w);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv ) 
{
    runTest( argc, argv);
    return EXIT_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////
//! Run a scan test for CUDA
////////////////////////////////////////////////////////////////////////////////
void runTest( int argc, char** argv )
{
    float device_time;
    float host_time;

    unsigned int world_size = WORLD_WIDTH * WORLD_HEIGHT;
    unsigned int mem_size = sizeof(unsigned) * world_size;

    // randomly initialize the world in host memory
    int behive[6][2]= {{0,1},{0,2},{1,0},{1,3},{2,1},{2,2}};
    // int glider[5][2]= {{0,1},{1,2},{2,0},{2,1},{2,2}};
    unsigned *h_world = (unsigned*) malloc (mem_size);
    customInit(h_world, behive, 6);

    unsigned int timer;
    CUT_SAFE_CALL(cutCreateTimer(&timer));

    // compute reference solution
    unsigned *gold_world = (unsigned*) malloc (mem_size);
    cutStartTimer(timer);
    computeGoldSeq(gold_world, h_world, WORLD_WIDTH, WORLD_HEIGHT, ITERATIONS);
    cutStopTimer(timer);
    printf("\n\n**===-------------------------------------------------===**\n");
    printf("Processing %d x %d world ...\n", WORLD_WIDTH, WORLD_HEIGHT);
    printf("Host CPU Processing time: %f (ms)\n", cutGetTimerValue(timer));
    print_matrix(gold_world, WORLD_HEIGHT, WORLD_WIDTH);
    host_time = cutGetTimerValue(timer);
    CUT_SAFE_CALL(cutDeleteTimer(timer));
}

void randomInit( unsigned int* world )
{
    for( unsigned int i = 0; i < WORLD_WIDTH * WORLD_HEIGHT; ++i) 
    {
        world[i] = (int)(rand() % 2);
    }
}

void customInit(unsigned int* world, int (*coords)[2], int len)
{
    // if world width and height is less than 5, initialize with random values
    if (WORLD_WIDTH < 5 || WORLD_HEIGHT < 5) {
        randomInit(world);
        return;
    }
    // zero out the world
    for( unsigned int i = 0; i < WORLD_WIDTH * WORLD_HEIGHT; ++i) 
    {
        world[i] = 0;
    }
    // initialize the world with the given coordinates
    // printf("len %d", len);
    for (int i = 0; i < len; i++) {
        world[coords[i][0] * WORLD_WIDTH + coords[i][1]] = 1;
    }
}


