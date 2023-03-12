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

// previous kernels and implementations

__device__ inline int dToShIndex(int x) {
    return (x + 1) / BYTES_PER_THREAD - (blockIdx.x * blockDim.x);
}

__global__ void lookup_with_shared_mem(unsigned char* d_world_in, unsigned char* d_world_out,  unsigned char* lookup_table,
    int const width, int const height)
{
    int tx = threadIdx.x;  
    int bx = blockIdx.x; 

    int world_size = height * width;
    int tid = bx * blockDim.x + tx;

    // int sh_size = 2 * BLOCK_SIZE; // start and end cell is shared the most
    int sh_width = 2 * width / BYTES_PER_THREAD; // 2 * threads per row
    __shared__ unsigned char sh_world[2 * BLOCK_SIZE]; // 2 * number of threads in a block

    // __shared__ unsigned int sh_lookup[2 * BLOCK_SIZE];

    // load first and last element of the BYTES_PER_THREAD cells into shared memory
    // these are the cells that are shared the most
    sh_world[2 * tx] = d_world_in[tid * BYTES_PER_THREAD];
    // sh_lookup[2 * tx] = tid * BYTES_PER_THREAD;
    // if (tid < 20) {
    //     printf("sh index = %d, d index = %d\n", 2 * tx, tid * BYTES_PER_THREAD);
    // }
    sh_world[2 * tx + 1] = d_world_in[(tid + 1) * BYTES_PER_THREAD - 1];
    // sh_lookup[2 * tx + 1] = (tid + 1) * BYTES_PER_THREAD - 1;
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

        uint sh_x = (tx * 2 + sh_width - 1) % sh_width;
        uint sh_y = (tx * 2 / sh_width) * sh_width;
        uint sh_x_next = (sh_x + 1) % sh_width;

        uint data0;
        uint data1 = (uint)sh_world[sh_x + sh_y] << 8 | sh_world[sh_x_next + sh_y];
        uint data2;

        // can't use shared memory for the top and bottom edge of the block
            data0 = (uint) d_world_in[yUp + x]  << 8 |  d_world_in[yUp + x_next];
            data2 = (uint) d_world_in[yDown + x]  << 8 |  d_world_in[yDown + x_next];

        x = x_next;
        sh_x = sh_x_next;

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

            // evaluates entire row of 8 cells
            d_world_out[currentState + y] = (lookup_table[HighFourBitStates] << 4) | lookup_table[LowFourBitStates];
        }

        uint currentState = x; // current cell
        x = (x + 1) % width; // load in the 3rd, 4th, 5th, .... byte
        sh_x = (sh_x + 1) % width;

        data1 = (data1 << 8) | (uint) sh_world[sh_y + sh_x]; // the cell to the right and down

        data0 = (data0 << 8) | (uint) d_world_in[yUp + x];
        data2 = (data2 << 8) | (uint) d_world_in[yDown + x];

        // encodes 6 * 3 block into one 18 bit number to pass in as a key to the lookup table
        uint HighFourBitStates = ((data0 & 0x1F800) << 1) | ((data1 & 0x1F800) >> 5) | ((data2 & 0x1F800) >> 11);
        uint LowFourBitStates = ((data0 & 0x1F80) << 5) | ((data1 & 0x1F80) >> 1) | ((data2 & 0x1F80) >> 7);

        d_world_out[currentState + y] = (lookup_table[HighFourBitStates] << 4) | lookup_table[LowFourBitStates];
        

        currentState = x; // current cell
        x = (x + 1) % width; // load in the 3rd, 4th, 5th, .... byte
        sh_x = (sh_x + 1) % sh_width;
        data1 = (data1 << 8) | (uint) sh_world[sh_y + sh_x]; // the cell to the right and down

        data0 = (data0 << 8) | (uint) d_world_in[yUp + x];
        data2 = (data2 << 8) | (uint) d_world_in[yDown + x];
        
        // encodes 6 * 3 block into one 18 bit number to pass in as a key to the lookup table
        HighFourBitStates = ((data0 & 0x1F800) << 1) | ((data1 & 0x1F800) >> 5) | ((data2 & 0x1F800) >> 11);
        LowFourBitStates = ((data0 & 0x1F80) << 5) | ((data1 & 0x1F80) >> 1) | ((data2 & 0x1F80) >> 7);

        d_world_out[currentState + y] = (lookup_table[HighFourBitStates] << 4) | lookup_table[LowFourBitStates];
    }
} 

__global__ void basic_lookup_kernel(unsigned char* d_world_in, unsigned char* d_world_out,  unsigned char* lookup_table,
    int const width, int const height)
{
    int tx = threadIdx.x;  
    int bx = blockIdx.x; 

    int world_size = height * width;
    int tid = bx * blockDim.x + tx;

    for (int i = tid * BYTES_PER_THREAD; i < world_size; i += blockDim.x * gridDim.x * BYTES_PER_THREAD)
    {
        // ** Fetch 2 bytes and the rows above and below  as data0,1,2**
        
        uint x = (i + width - 1) % width; // evaluates x - 1 first 
        uint y = (i / width) * width; // y: y offest of the cell
        uint yUp = (y + world_size - width) % world_size; // yUp: y offset of the cell above
        uint yDown = (y + width) % world_size; // yDown: y offset of the cell below
        // 3 integers to hold the 3 rows
        // load in the first byte
        uint data0 = (uint)d_world_in[yUp + x]  << 8;
        uint data1 = (uint)d_world_in[y + x]  << 8;
        uint data2 = (uint)d_world_in[yDown + x]  << 8;

        // load in the second byte
        x = (x + 1) % width; // increment x to the next cell
        data0 |= (uint) d_world_in[yUp + x]; // the cell to the right and up
        data1 |= (uint) d_world_in[y + x]; // the cell to the right and down
        data2 |= (uint) d_world_in[yDown + x]; // the cell to the right and down


        for (uint j = 0; j < BYTES_PER_THREAD; j++)
        {
            uint currentState = x; // current cell
            x = (x + 1) % width; // load in the 3rd, 4th, 5th, .... byte
            data0 = (data0 << 8) | (uint) d_world_in[yUp + x]; // the cell to the right and up
            data1 = (data1 << 8) | (uint) d_world_in[y + x]; // the cell to the right and down
            data2 = (data2 << 8) | (uint) d_world_in[yDown + x]; // the cell to the right and down
            
            // encodes 6 * 3 block into one 18 bit number to pass in as a key to the lookup table
            uint HighFourBitStates = ((data0 & 0x1F800) << 1) | ((data1 & 0x1F800) >> 5) | ((data2 & 0x1F800) >> 11);
			uint LowFourBitStates = ((data0 & 0x1F80) << 5) | ((data1 & 0x1F80) >> 1) | ((data2 & 0x1F80) >> 7);
            d_world_out[currentState + y] = (lookup_table[HighFourBitStates] << 4) | lookup_table[LowFourBitStates];
        }
    }
} 

__global__ void basic_bit_representation(unsigned char* d_world_in, unsigned char* d_world_out, int height, int width)
{
    int tx = threadIdx.x; 
    int bx = blockIdx.x; 

    int world_size = height * width;
    int tid = bx * blockDim.x + tx;

    // tid is indx of the cell in the world
    // i is the index of every 2 cells, incremented essentially by world size in current setup
    for (int i = tid * BYTES_PER_THREAD; i < world_size; i += blockDim.x * gridDim.x * BYTES_PER_THREAD)
    {
        // ** Fetch 2 bytes and the rows above and below  as data0,1,2**
        
        uint x = (i + width - 1) % width; // evaluates x - 1 first 
        uint y = i / width * width; // y: y offest of the cell
        uint yUp = (y + world_size - width) % world_size; // yUp: y offset of the cell above
        uint yDown = (y + width) % world_size; // yDown: y offset of the cell below
        // 3 integers to hold the 3 rows
        // load in the first byte
        uint data0 = (uint)d_world_in[yUp + x]  << 16;
        uint data1 = (uint)d_world_in[y + x]  << 16;
        uint data2 = (uint)d_world_in[yDown + x]  << 16;

        // load in the second byte
        x = (x + 1) % width; // increment x to the next cell
        data0 |= (uint) d_world_in[yUp + x] << 8; // the cell to the right and up
        data1 |= (uint) d_world_in[y + x] << 8; // the cell to the right and down
        data2 |= (uint) d_world_in[yDown + x] << 8; // the cell to the right and down

        for (uint j = 0; j < BYTES_PER_THREAD; j++)
        {
            uint currentState = x; // current cell
            x = (x + 1) % width; // load in the 3rd, 4th, 5th, .... byte
            data0 |= (uint) d_world_in[yUp + x]; // the cell to the right and up
            data1 |= (uint) d_world_in[y + x]; // the cell to the right and down
            data2 |= (uint) d_world_in[yDown + x]; // the cell to the right and down

            uint result = 0;
            for (uint k = 0; k < 8; k++) // loop through each bit of the char
            {
                uint neighbours = (data0 & 0x14000) + (data1 & 0x14000) + (data2 & 0x14000);
                neighbours >>= 14;
                neighbours = (neighbours & 0x3) + (neighbours >> 2) + ((data0 >> 15) & 0x1u) + ((data2 >> 15) & 0x1u);
                
                // if (neighbours > 0 && tid == 1)
                //     printf("\nidx: %d |\t neighbours: %d", j * 8 + k, neighbours);

                result = result << 1 | (neighbours == 3 || (neighbours == 2 && (data1 & 0x8000u)) ? 1u : 0u);

                data0 <<= 1;
                data1 <<= 1;
                data2 <<= 1;
            }
            d_world_out[currentState + y] = result;
        }
    }
} 

__global__ void baseline_gpu(unsigned* d_world_in, unsigned* d_world_out, int height, int width)
{
    int tx = threadIdx.x; 
    int bx = blockIdx.x; 

    int world_size = height * width;
    int tid = bx * blockDim.x + tx;

    for (uint i = tid; i < world_size; i += blockDim.x * gridDim.x)
    {
        int n = 0;
        for (int x = -1; x <= 1; x++) {
            for (int y = -1; y <= 1; y++) {
                if (x == 0 && y == 0) continue;
                int xAbs = (i % width) + x;
                int yAbs = (i - (i % width)) + y * width;
                if (xAbs < 0) xAbs += width;
                if (xAbs >= width) xAbs -= width;
                if (yAbs < 0) yAbs += world_size;
                if (yAbs >= world_size) yAbs -= world_size;
                n += d_world_in[yAbs + xAbs];
            }
        }
        
        d_world_out[i] = (n == 3 || (n == 2 && d_world_in[i]));
    }
} 


#endif // _CONWAY_CU_
