#ifndef _CONWAY_CU_
#define _CONWAY_CU_

#include <assert.h>

////////////////////////////////////////////////////////////////////////////////
extern "C" 
void printBinary(unsigned n);

extern "C" 
void printMatrix(unsigned *u, int h, int w);
////////////////////////////////////////////////////////////////////////////////

# define WORLD_WIDTH 16384
# define WORLD_HEIGHT 16384
# define ITERATIONS 100
      
# define VERBOSE false  
# define IS_RAND true  

#define BLOCK_SIZE 128
#define BYTES_PER_THREAD 16

__global__ void conway_kernel(unsigned* d_world_in, unsigned* d_world_out,  unsigned char* lookup_table,
    int const width, int const height)
{
    int tx = threadIdx.x;  
    int bx = blockIdx.x; 

    int world_size = height * width;
    int tid = bx * blockDim.x + tx;

    // tid is indx of the cell in the world
    // i is the index of every 2 cells, incremented essentially by world size in current setup
    for (int i = tid * BYTES_PER_THREAD / 4; i < world_size; i += blockDim.x * gridDim.x * BYTES_PER_THREAD * 8)
    {
        // ** Fetch 2 bytes and the rows above and below  as data0,1,2**
        
        uint x = (i + width - 1) % width; // evaluates x - 1 first 
        uint y = (i / width) * width; // y: y offest of the cell
        uint yUp = (y + world_size - width) % world_size; // yUp: y offset of the cell above
        uint yDown = (y + width) % world_size; // yDown: y offset of the cell below
        // 3 integers to hold the 3 rows
        // load in the first byte
        uint64_t data0 = (uint64_t) d_world_in[yUp + x]  << 32;
        uint64_t data1 = (uint64_t) d_world_in[y + x]  << 32;
        uint64_t data2 = (uint64_t) d_world_in[yDown + x]  << 32;

        x = (x + 1) % width; // load in first uint
        data0 |= d_world_in[yUp + x];
        data1 |= d_world_in[y + x];
        data2 |= d_world_in[yDown + x];

        for (uint j = 0; j < BYTES_PER_THREAD / 4; j++)
        {
            uint currentState = x; // current cell
            x = (x + 1) % width; // load in the 3rd, 4th, 5th, .... byte
            uint newData0 = d_world_in[yUp + x];
            uint newData1 = d_world_in[y + x];
            uint newData2 = d_world_in[yDown + x];

            // load in the left most byte of the new int
            data0 = (data0 << 8) | (newData0 >> 24); // the cell to the right and up
            data1 = (data1 << 8) | (newData1 >> 24);// the cell to the right and down
            data2 = (data2 << 8) | (newData2 >> 24); // the cell to the right and down

            // get the updated state of the cells in the first uint loaded in
            // the bytes we need are at: A: 8, B: 16, C: 24, D: 32
            uint highA = (((data0 & 0x1F800000000) >> 23) | ((data1 & 0x1F800000000) >> 29) | ((data2 & 0x1F800000000) >> 35));
            uint lowA = (((data0 & 0x1F80000000) >> 19) | ((data1 & 0x1F80000000) >> 25) | ((data2 & 0x1F80000000) >> 31));
            uint highB = (((data0 & 0x1F8000000) >> 15) | ((data1 & 0x1F8000000) >> 21) | ((data2 & 0x1F8000000) >> 27));
            uint lowB = (((data0 & 0x1F800000) >> 11) | ((data1 & 0x1F800000) >> 17) | ((data2 & 0x1F800000) >> 23));
            uint highC = (((data0 & 0x1F80000) >> 7) | ((data1 & 0x1F80000) >> 13) | ((data2 & 0x1F80000) >> 19));
            uint lowC = (((data0 & 0x1F8000) >> 3) | ((data1 & 0x1F8000) >> 9) | ((data2 & 0x1F8000) >> 15));
            uint highD = (((data0 & 0x1F800) << 1) | ((data1 & 0x1F800) >> 5) | ((data2 & 0x1F800) >> 11));
            uint lowD = (((data0 & 0x1F80) << 5) | ((data1 & 0x1F80) >> 1) | ((data2 & 0x1F80) >> 7));

            // get the updated state of the cells in the second uint loaded in
            d_world_out[currentState + y] = (
                (lookup_table[highA] << 28) | (lookup_table[lowA] << 24) |
                (lookup_table[highB] << 20) | (lookup_table[lowB] << 16) |
                (lookup_table[highC] << 12) | (lookup_table[lowC] << 8) |
                (lookup_table[highD] << 4) | (lookup_table[lowD])
            );

            data0 = data0 << 24 | newData0;
            data1 = data1 << 24 | newData1;
            data2 = data2 << 24 | newData2;
        }

    }
} 

void runConwayKernel(unsigned ** d_world_in, unsigned ** d_world_out, unsigned char* lookup_table,
    const int width, const int height, int iterations)
{   
    size_t numBlocks = (height * width) / 32 / 4 / BLOCK_SIZE;
    dim3 dimBlock(BLOCK_SIZE, 1, 1);
    dim3 dimGrid(numBlocks, 1, 1);
    if (iterations > 1)
    {
        printf(" - block size:\t\t%d\n - bytes per thread:\t%d\n", BLOCK_SIZE, BYTES_PER_THREAD);
    }
    for (int i = 0; i < iterations; i++)
    {
        conway_kernel<<<dimGrid, dimBlock>>>(*d_world_in, *d_world_out, lookup_table, width / 32, height);
        std::swap(*d_world_in, *d_world_out);
    }
}

#endif // _CONWAY_CU_
