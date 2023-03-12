#ifdef _WIN32
#  define NOMINMAX 
#endif

#include <stdlib.h>
#include <stdio.h>     
#include <string.h>  
#include <math.h>    
#include <cutil.h>  
 
#include <GL/glew.h>    
#include <GL/glut.h> 
  
#include <conway_kernel.cu>

/*
    works for:
        4096 x 4096 
        4096 x 2048
        8192 x 8192
        16384 x 16384
        256 x 64
    fails for:
        4096 x 1024 
        2048 x 2048 .. 32 
        4096 x 512 ... 16
        1024 x 1024 ... 64
        512 x 512
        256 x 256  
*/
 
// questions
// 1. why not bigger worlds?
// 2. 
    
//////////////////////////////////////////////////////////////////////////////// 
// main test routine    
void init();    
void display(); 
void runTest( int argc, char** argv );  

void randomInit( unsigned int* world );
void customInit(unsigned int* world, int (*coords)[2], int len);
/////////////////////////////////////////////////////////////////////////////////
// conway_gold.cpp
extern "C"  
void computeGoldSeq(  unsigned* reference, unsigned* idata, int width, int height, int iterations); 

extern "C"
void computeLookupTable(unsigned char* lookup_table, uint dimX, uint dimY);

extern "C" 
unsigned int compare( const unsigned* reference, unsigned* data, const unsigned int len, const bool verbose); 

extern "C" 
void printMatrix(unsigned *u, int h, int w);
//////////////////////////////////////////////////////////////////////////////// 
// bit_utils.cpp   
extern "C" 
void printBinary(unsigned n);

extern "C" 
void bitPerCellEncode(unsigned *in, unsigned  *out, int width, int height); 

extern "C"
void bitPerCellDecode(unsigned *in, unsigned *out, int width, int height);  
//////////////////////////////////////////////////////////////////////////////// 
// Program main
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv ) 
{
    // glutInit(&argc, argv); 
    // glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    // glutInitWindowSize(WORLD_WIDTH, WORLD_HEIGHT);
    // glutCreateWindow("Conway's Game of Life");

    // glewInit();

    // glutDisplayFunc(display);
    
    // glutMainLoop();

    // return 0;

    runTest( argc, argv);
    return EXIT_SUCCESS;
}

void display()
{
    glClear(GL_COLOR_BUFFER_BIT);
    
    glBegin(GL_QUADS);
        glColor3f(1.0, 0.0, 0.0);
        glVertex2f(-0.5, -0.5);
        glVertex2f(-0.5, 0.5);
        glVertex2f(0.5, 0.5);
        glVertex2f(0.5, -0.5);
    glEnd();

    glutSwapBuffers();
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
    unsigned int bit_mem_size = sizeof(unsigned) * (world_size / 32); // pad with + 1 if not divisible by 8

    // randomly initialize the world in host memory
    // int behive[6][2]= {{0,1},{0,2},{1,0},{1,3},{2,1},{2,2}};
    // int glider[5][2]= {{0,1},{1,2},{2,0},{2,1},{2,2}};
    // int pulsar[48][2] = {{2,4}, {2,5}, {2,6}, {2,10}, {2,11}, {2,12}, {4,2}, {4,7}, {4,9}, {4,14}, {5,2}, {5,7}, {5,9}, {5,14}, {6,2}, {6,7}, {6,9}, {6,14}, {7,4}, {7,5}, {7,6}, {7,10}, {7,11}, {7,12}, {9,4}, {9,5}, {9,6}, {9,10}, {9,11}, {9,12}, {10,2}, {10,7}, {10,9}, {10,14}, {11,2}, {11,7}, {11,9}, {11,14}, {12,2}, {12,7}, {12,9}, {12,14}, {14,4}, {14,5}, {14,6}, {14,10}, {14,11}, {14,12}};
    // int glider_gun[36][2] = {{0,4},{0,5},{1,4},{1,5},{10,4},{10,5},{10,6},{11,3},{11,7},{12,2},{12,8},{13,2},{13,8},{14,5},{15,3},{15,7},{16,4},{16,5},{16,6},{17,5},{20,2},{20,3},{20,4},{21,2},{21,3},{21,4},{22,1},{22,5},{24,0},{24,1},{24,5},{24,6},{34,2},{34,3},{35,2},{35,3}};
    // int bigOscillator[92][2] = {{1,0},{2,0},{22,0},{23,0},{1,1},{2,1},{21,1},{23,1},{24,1},{0,2},{1,2},{2,2},{20,2},{21,2},{24,2},{25,2},{0,3},{1,3},{2,3},{20,3},{22,3},{24,3},{25,3},{0,4},{1,4},{2,4},{3,4},{4,4},{19,4},{20,4},{22,4},{24,4},{25,4},{26,4},{4,5},{19,5},{20,5},{4,6},{5,6},{6,6},{18,6},{19,6},{6,7},{7,7},{8,7},{16,7},{17,7},{18,7},{8,8},{9,8},{15,8},{16,8},{9,9},{10,9},{14,9},{15,9},{10,10},{11,10},{12,10},{13,10}};
    unsigned *h_world = (unsigned*) malloc (mem_size);
    // customInit(h_world, pulsar, 48); 
    randomInit(h_world); 

    if (VERBOSE) {
        printf("initial world: \n");
        printMatrix(h_world, WORLD_HEIGHT, WORLD_WIDTH);
    }

    unsigned int timer;  
    CUT_SAFE_CALL(cutCreateTimer(&timer));

    // compute reference solution using sequential cpu
    unsigned *gold_world = (unsigned*) malloc (mem_size);
    cutStartTimer(timer); 
    computeGoldSeq(gold_world, h_world, WORLD_WIDTH, WORLD_HEIGHT, ITERATIONS);
    cutStopTimer(timer);
     
    printf("\033[1;33mProcessing %d x %d world for %d iterations\033[0m\n", WORLD_WIDTH, WORLD_HEIGHT, ITERATIONS);
    // printf("HOST CPU Processing time: %f (ms)\n", cutGetTimerValue(timer));
      
    if (VERBOSE) {
        printf("cpu computed world: \n");
        printMatrix(gold_world, WORLD_HEIGHT, WORLD_WIDTH); 
    }

    host_time = cutGetTimerValue(timer);
    CUT_SAFE_CALL(cutDeleteTimer(timer));

    // **===----------------- Allocate device data structures -----------===**
    unsigned *d_world_in; 
    unsigned *d_world_out;
    

    // encode the world into a bit array
    unsigned *h_world_bits = (unsigned *) malloc (bit_mem_size);
    bitPerCellEncode(h_world, h_world_bits, WORLD_WIDTH, WORLD_HEIGHT);

    // unsigned *temp_world = (unsigned*) malloc (mem_size);
    // bitPerCellDecode(h_world_bits, temp_world, WORLD_WIDTH, WORLD_HEIGHT);
    // printf("bits world [0]:  %d\n", h_world_bits[0]); // 64 for glider
    // printf("bits world [1]:  %d\n", h_world_bits[1]); // 142 for glider
    // printMatrix(temp_world, WORLD_HEIGHT, WORLD_WIDTH);  
    // free(temp_world);
 
    uint lookup_x = 6; 
    uint lookup_y = 3;
    uint lookup_table_size = 1 << (lookup_x * lookup_y) * sizeof(unsigned char);
    unsigned char *h_lookup_table = (unsigned char*) malloc (lookup_table_size); 
    unsigned char *d_lookup_table;

    computeLookupTable(h_lookup_table, lookup_x, lookup_y);
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_lookup_table, lookup_table_size));
    CUDA_SAFE_CALL(cudaMemcpy(d_lookup_table, h_lookup_table, lookup_table_size, cudaMemcpyHostToDevice));

    printf(" - lookup table:\t%d x %d  (approx. %d kB)\n", lookup_x, lookup_y, lookup_table_size/1024);

    CUDA_SAFE_CALL(cudaMalloc((void**) &d_world_in, bit_mem_size));
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_world_out, bit_mem_size)); 
    // copy host memory to device input array
    CUDA_SAFE_CALL(cudaMemcpy(d_world_in, h_world_bits, bit_mem_size, cudaMemcpyHostToDevice));
    // initialize all the other device arrays to be safe
    CUDA_SAFE_CALL(cudaMemcpy(d_world_out, h_world_bits, bit_mem_size, cudaMemcpyHostToDevice) );

    // **===----------------- Launch the device computation ----------------===**   
    // run once to remove startup overhead
    runConwayKernel(&d_world_in, &d_world_out, d_lookup_table, WORLD_WIDTH, WORLD_HEIGHT, 1);

    CUT_SAFE_CALL(cutCreateTimer(&timer));
    cutStartTimer(timer);
    // run the kernel 
    runConwayKernel(&d_world_in, &d_world_out, d_lookup_table, WORLD_WIDTH, WORLD_HEIGHT, ITERATIONS);
    CUDA_SAFE_CALL( cudaDeviceSynchronize() );

    cutStopTimer(timer);
    device_time = cutGetTimerValue(timer);  
    printf("**===-------------------------------------------------===**\n");
    printf("\tHOST CPU Processing time: %f (ms)\n", host_time);
    printf("\tCUDA GPU Processing time: %f (ms)\n", device_time);
    printf("\tSpeedup: %fX\n", host_time/device_time);     
    printf("**===-------------------------------------------------===**\n");

    
    // **===-------- Deallocate data structure  -----------===**
    CUDA_SAFE_CALL(cudaMemcpy(h_world_bits, d_world_out, bit_mem_size, cudaMemcpyDeviceToHost));
 
    // decode the world from the bit array
    bitPerCellDecode(h_world_bits, h_world, WORLD_WIDTH, WORLD_HEIGHT);

    if (VERBOSE) {
        printf("gpu computed world: \n");
        printMatrix(h_world, WORLD_HEIGHT, WORLD_WIDTH);  
    }
 
    unsigned int result = compare(gold_world, h_world, world_size, VERBOSE);
    printf("Test %s\n", (1 == result) ? "\033[1;32mPASSED \033[0m" : "\033[1;31mFAILED \033[0m");  

    if (VERBOSE && result == 0){
        printf("world with errors: \n");
        printMatrix(h_world, WORLD_HEIGHT, WORLD_WIDTH);     
    }

    CUT_SAFE_CALL(cutDeleteTimer(timer));
    free(h_world);  
    free(h_world_bits);
    free(gold_world); 
    CUDA_SAFE_CALL(cudaFree(d_world_in)); 
    CUDA_SAFE_CALL(cudaFree(d_world_out));
} 
 
void randomInit( unsigned int* world )  
{ 
    if (IS_RAND) {
        srand( time(NULL) );
    } else { 
        srand( 22223 ) ;
    }
    for( unsigned int i = 0; i < WORLD_WIDTH * WORLD_HEIGHT; ++i) 
    {    
        world[i] = (int)(rand() % 2);
    } 
}

void customInit(unsigned int* world, int (*coords)[2], int len) 
{
    // // if world width and height is less than 5, initialize with random values
    // if (WORLD_WIDTH < len || WORLD_HEIGHT < len) {
    //     randomInit(world); 
    //     return;
    // }
    // zero out the world 
    for( unsigned int i = 0; i < WORLD_WIDTH * WORLD_HEIGHT; ++i) 
    {
        world[i] = 0;
    } 
    // initialize the world with the given coordinates
    // printf("len %d", len);
    for (int i = 0; i < len; i++) {
        if (coords[i][0] >= WORLD_HEIGHT || coords[i][1] >= WORLD_WIDTH) {
            printf("Error: coordinates out of bounds\n");
            exit(1);
        }
        world[coords[i][0] * WORLD_WIDTH + coords[i][1]] = 1;
        // printf("coords %d %d\n", coords[i][0], coords[i][1]);
    }
}

