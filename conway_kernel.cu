#ifndef _CONWAY_CU_
#define _CONWAY_CU_

#include <assert.h>

////////////////////////////////////////////////////////////////////////////////
extern "C" 
void printBinary(unsigned n);

extern "C" 
void printMatrix(unsigned *u, int h, int w);
////////////////////////////////////////////////////////////////////////////////
# define WORLD_WIDTH 4096
# define WORLD_HEIGHT 4096 
# define ITERATIONS 100
      
# define VERBOSE false  
# define IS_RAND false     

#define BLOCK_SIZE 256
#define BYTES_PER_THREAD 16

// TODO: shared mem edges
// TODO: 64 word int
// TODO: 2 layers + 4 * 6 lookup

__device__ inline int dToShIndex(int x) {
    return (x + 1) / BYTES_PER_THREAD - (blockIdx.x * blockDim.x);
}

__global__ void conway_kernel(unsigned char* d_world_in, unsigned char* d_world_out,  unsigned char* lookup_table,
    int const width, int const height)
{
    int tx = threadIdx.x;  
    int bx = blockIdx.x; 

    int world_size = height * width;
    int tid = bx * blockDim.x + tx;

    int sh_size = 2 * BLOCK_SIZE; // start and end cell is shared the most
    int sh_width = 2 * width / BYTES_PER_THREAD; // 2 * threads per row
    __shared__ unsigned char sh_world[2 * BLOCK_SIZE]; // 2 * number of threads in a block

    __shared__ unsigned int sh_lookup[2 * BLOCK_SIZE];

    // load first and last element of the BYTES_PER_THREAD cells into shared memory
    // these are the cells that are shared the most
    sh_world[2 * tx] = d_world_in[tid * BYTES_PER_THREAD];
    sh_lookup[2 * tx] = tid * BYTES_PER_THREAD;
    // if (tid < 20) {
    //     printf("sh index = %d, d index = %d\n", 2 * tx, tid * BYTES_PER_THREAD);
    // }
    sh_world[2 * tx + 1] = d_world_in[(tid + 1) * BYTES_PER_THREAD - 1];
    sh_lookup[2 * tx + 1] = (tid + 1) * BYTES_PER_THREAD - 1;
    // if (tid < 20) {
    //     printf("sh index 1 = %d, d index 1 = %d\n", 2 * tx + 1, (tid + 1) * BYTES_PER_THREAD - 1);
    // }

    __syncthreads();

    // tid is indx of the cell in the world
    // i is the index of every 2 cells, incremented essentially by world size in current setup
    for (int i = tid * BYTES_PER_THREAD; i < world_size; i += blockDim.x * gridDim.x * BYTES_PER_THREAD)
    {
        // ** Fetch 2 bytes and the rows above and below  as data0,1,2**
        
        uint x = (i + width - 1) % width; // evaluates x - 1 first 
        uint y = (i / width) * width; // y: y offest of the cell
        uint yUp = (y + world_size - width) % world_size; // yUp: y offset of the cell above
        uint yDown = (y + width) % world_size; // yDown: y offset of the cell below
        uint x_next = (x + 1) % width; // x_next: x offset current cell
        // 3 integers to hold the 3 rows
        // load in the first byte
        // uint data0 = (uint)d_world_in[yUp + x]  << 8;
        // uint data1 = (uint)d_world_in[y + x]  << 8;
        // uint data2 = (uint)d_world_in[yDown + x]  << 8;

        uint sh_x = (tx * 2 + sh_width - 1) % sh_width;
        uint sh_y = (tx * 2 / sh_width) * sh_width;
        uint sh_yUp = (sh_y + sh_size - sh_width) % sh_size;
        uint sh_yDown = (sh_y + sh_width) % sh_size;
        uint sh_x_next = (sh_x + 1) % sh_width;

        uint data0;
        uint data1 = (uint)sh_world[sh_x + sh_y] << 8 | sh_world[sh_x_next + sh_y];
        uint data2;
        // uint data22 = (uint)d_world_in[yDown + x]  << 8;

        // can't use shared memory for the top and bottom edge of the block
        if (tx <= sh_width) { // works if do sh_width / 2 here but make sure its 32 for divergence
            data0 = (uint) d_world_in[yUp + x]  << 8 |  d_world_in[yUp + x_next];
        } else {
            data0 = (uint) sh_world[sh_x + sh_yUp] << 8 |  sh_world[sh_x_next + sh_yUp];
        }

        if (tx >= BLOCK_SIZE - sh_width) { // works if do / 2 here but make sure its 32 for divergence
            data2 = (uint) d_world_in[yDown + x]  << 8 |  d_world_in[yDown + x_next];
        } else {
            data2 = (uint) sh_world[sh_x + sh_yDown] << 8 | sh_world[sh_x_next + sh_yDown];
        }

        x = x_next;
        sh_x = sh_x_next;

        // load in the second byte
        // x = (x + 1) % width; // increment x to the next cell
        // data0 |= (uint) d_world_in[yUp + x]; // the cell to the right and up
        // data1 |= (uint) d_world_in[y + x]; // the cell to the right and down
        // data2 |= (uint) d_world_in[yDown + x]; // the cell to the right and down

        // if (tid < 256) {
        //     if (data22 != data2) {
        //         printf("tid = %d, shid = %d, x = %d, y = %d, yUp = %d, yDown = %d\n", tid, sh_x + sh_yDown, sh_x, sh_y, sh_yUp, sh_yDown);
        //         printf("error at tid %d, incorrect: %d, correct: %d with y %d x %d\n", tid, sh_lookup[sh_x + sh_yDown], y + x, yDown,x);
        //                     //     for(int i = 31; i >= 0; i--) {
        //     //         if (i % 8 == 7)
        //     //             printf(" ");
        //     //         printf("%d", (data1 >> i) & 1);
        //     //     }
        //     //     printf("\n");
        //     //     for(int i = 31; i >= 0; i--) {
        //     //         if (i % 8 == 7)
        //     //             printf(" ");
        //     //         printf("%d", (data2 >> i) & 1);
        //     //     }
        //     //     printf("\n");
        //     }
        // }

        for (uint j = 0; j < BYTES_PER_THREAD - 2; j++)
        {
            uint currentState = x; // current cell
            x = (x + 1) % width; // load in the 3rd, 4th, 5th, .... byte
            data0 = (data0 << 8) | (uint) d_world_in[yUp + x]; // the cell to the right and up
            data1 = (data1 << 8) | (uint) d_world_in[y + x]; // the cell to the right and down
            data2 = (data2 << 8) | (uint) d_world_in[yDown + x]; // the cell to the right and down
            
            // encodes 6 * 3 block into one 18 bit number to pass in as a key to the lookup table
            uint HighFourBitStates = ((data0 & 0x1F800) << 1) | ((data1 & 0x1F800) >> 5) | ((data2 & 0x1F800) >> 11);
			uint LowFourBitStates = ((data0 & 0x1F80) << 5) | ((data1 & 0x1F80) >> 1) | ((data2 & 0x1F80) >> 7);

            // if (tid == 9) {
            //     printf("\nx = %d, y = %d, yUp = %d, yDown = %d\n", x, y, yUp, yDown);
            //     printf("printing data 0 1 2 \n");
            //     for(int i = 31; i >= 0; i--) {
            //         if (i % 8 == 7)
            //             printf(" ");
            //         printf("%d", (data0 >> i) & 1);
            //     }
            //     printf("\n");
            //     for(int i = 31; i >= 0; i--) {
            //         if (i % 8 == 7)
            //             printf(" ");
            //         printf("%d", (data1 >> i) & 1);
            //     }
            //     printf("\n");
            //     for(int i = 31; i >= 0; i--) {
            //         if (i % 8 == 7)
            //             printf(" ");
            //         printf("%d", (data2 >> i) & 1);
            //     }
            //     printf("\n");
            // }

            // evaluates entire row of 8 cells
            d_world_out[currentState + y] = (lookup_table[HighFourBitStates] << 4) | lookup_table[LowFourBitStates];

            // if (tid == 9) {
            //     printf("printing high \n");
            //     for(int i = 31; i >= 0; i--) {
            //         if (i % 8 == 7)
            //             printf(" ");
            //         printf("%d", (HighFourBitStates >> i) & 1);
            //     }
            //     printf("\n");
            //     printf("printing low \n");
            //     for(int i = 31; i >= 0; i--) {
            //         if (i % 8 == 7)
            //             printf(" ");
            //         printf("%d", (LowFourBitStates >> i) & 1);
            //     }
            //     printf("\n");
            //     printf("printing result \n");
            //     for(int i = 31; i >= 0; i--) {
            //         if (i % 8 == 7)
            //             printf(" ");
            //         printf("%d", (d_world_out[currentState + y] >> i) & 1);
            //     }
            //     printf("\n");
            // }

        }

        uint currentState = x; // current cell
        x = (x + 1) % width; // load in the 3rd, 4th, 5th, .... byte
        sh_x = (sh_x + 1) % width;
        // data0 = (data0 << 8) | (uint) d_world_in[yUp + x]; // the cell to the right and up
        data1 = (data1 << 8) | (uint) sh_world[sh_y + sh_x]; // the cell to the right and down
        // data2 = (data2 << 8) | (uint) d_world_in[yDown + x]; // the cell to the right and down

        if (tx <= sh_width) { // works if do sh_width / 2 here but make sure its 32 for divergence
            data0 = (data0 << 8) | (uint) d_world_in[yUp + x];
        } else {
            data0 = (data0 << 8) | (uint) sh_world[sh_x + sh_yUp];
        }
        

        if (tx >= BLOCK_SIZE - sh_width) { // works if do / 2 here but make sure its 32 for divergence
            data2 = (data2 << 8) | (uint) d_world_in[yDown + x];
        } else {
            data2 = (data2 << 8) | (uint) sh_world[sh_yDown + sh_x];
        }
        
        // encodes 6 * 3 block into one 18 bit number to pass in as a key to the lookup table
        uint HighFourBitStates = ((data0 & 0x1F800) << 1) | ((data1 & 0x1F800) >> 5) | ((data2 & 0x1F800) >> 11);
        uint LowFourBitStates = ((data0 & 0x1F80) << 5) | ((data1 & 0x1F80) >> 1) | ((data2 & 0x1F80) >> 7);

        d_world_out[currentState + y] = (lookup_table[HighFourBitStates] << 4) | lookup_table[LowFourBitStates];
        

        currentState = x; // current cell
        x = (x + 1) % width; // load in the 3rd, 4th, 5th, .... byte
        sh_x = (sh_x + 1) % sh_width;
        // data0 = (data0 << 8) | (uint) d_world_in[yUp + x]; // the cell to the right and up
        data1 = (data1 << 8) | (uint) sh_world[sh_y + sh_x]; // the cell to the right and down
        // data2 = (data2 << 8) | (uint) d_world_in[yDown + x]; // the cell to the right and down

        if (tx <= sh_width) { // works if do sh_width / 2 here but make sure its 32 for divergence
            data0 = (data0 << 8) | (uint) d_world_in[yUp + x];
        } else {
            data0 = (data0 << 8) | (uint) sh_world[sh_x + sh_yUp];
        }
        

        if (tx >= BLOCK_SIZE - sh_width) { // works if do / 2 here but make sure its 32 for divergence
            data2 = (data2 << 8) | (uint) d_world_in[yDown + x];
        } else {
            data2 = (data2 << 8) | (uint) sh_world[sh_yDown + sh_x];
        }
        
        // encodes 6 * 3 block into one 18 bit number to pass in as a key to the lookup table
        HighFourBitStates = ((data0 & 0x1F800) << 1) | ((data1 & 0x1F800) >> 5) | ((data2 & 0x1F800) >> 11);
        LowFourBitStates = ((data0 & 0x1F80) << 5) | ((data1 & 0x1F80) >> 1) | ((data2 & 0x1F80) >> 7);

        d_world_out[currentState + y] = (lookup_table[HighFourBitStates] << 4) | lookup_table[LowFourBitStates];
    }
} 

void runConwayKernel(unsigned char** d_world_in, unsigned char** d_world_out, unsigned char* lookup_table,
    const int width, const int height, int iterations)
{   
    // TODO: handle case when things are not divisible by 8
    // may need to pad the matrix with the otherside of the matrix
    // assert(((height * width) / 8 / BYTES_PER_THREAD) % BLOCK_SIZE == 0);

    // each thread will process BYTES_PER_THREAD * 8 cells
    // each block will process contiguous BYTES_PER_THREAD * 8 * BLOCK_SIZE cells

    size_t numBlocks = (height * width) / 8 / BYTES_PER_THREAD / BLOCK_SIZE;
    dim3 dimBlock(BLOCK_SIZE, 1, 1);
    dim3 dimGrid(numBlocks, 1, 1);
    if (iterations > 1)
    {
        printf(" - block size:\t\t%d\n - bytes per thread:\t%d\n", BLOCK_SIZE, BYTES_PER_THREAD);
    }
    for (int i = 0; i < iterations; i++)
    {

        conway_kernel<<<dimGrid, dimBlock>>>(*d_world_in, *d_world_out, lookup_table, width / 8, height);
        std::swap(*d_world_in, *d_world_out);
        // printf("iteration %d\n", i + 1);
        // // print first 5 elements of d_world_in
        // for (int j = 0; j < 5; i++) {
        //     printf("%d ", *d_world_in[j]);
        // }
        // printf("\n");
    }
}

#endif // _CONWAY_CU_


    // for(int i = 0; i < height; i++) {
    //     for(int j = 0; j < width; j++) {
    //         printf("%d ", *d_world_out[i*(width) + j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");