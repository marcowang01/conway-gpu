#ifndef _CONWAY_CU_
#define _CONWAY_CU_

#include <assert.h>
#include <algorithm>

////////////////////////////////////////////////////////////////////////////////
extern "C" 
void printBinary(unsigned n);

extern "C" 
void printMatrix(unsigned *u, int h, int w);
////////////////////////////////////////////////////////////////////////////////
// total bytes per block:
#define BLOCK_SIZE 128
#define BYTES_PER_THREAD 16

// TODO: change back to using ints for better mem access? 
// TODO: change to using share memory

__global__ void conway_kernel(unsigned char* d_world_in, unsigned char* d_world_out,  unsigned char* lookup_table,
    int const width, int const height)
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
        }
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

__global__ void displayLifeKernel(const char* lifeData, uint worldWidth, uint worldHeight, uchar4* destination,
        int destWidth, int detHeight, int2 displacement, double zoomFactor, int multisample) {

    uint pixelId = blockIdx.x * blockDim.x + threadIdx.x;

    int x = (int)floor(((int)(pixelId % destWidth) - displacement.x) * zoomFactor);
    int y = (int)floor(((int)(pixelId / destWidth) - displacement.y) * zoomFactor);

    x = ((x % (int)worldWidth) + worldWidth) % worldWidth;
    y = ((y % (int)worldHeight) + worldHeight) % worldHeight;

    int value = 0;  // Start at value - 1.
    int increment = 255 / (multisample * multisample);

    for (int dy = 0; dy < multisample; ++dy) {
        int yAbs = (y + dy) * worldWidth;
        for (int dx = 0; dx < multisample; ++dx) {
            int xBucket = yAbs + x + dx;
            value += ((lifeData[xBucket >> 3] >> (7 - (xBucket & 0x7))) & 0x1) * increment;
        }
    }

    bool isNotOnBoundary = !(x == 0 || y == 0);

    destination[pixelId].x = isNotOnBoundary ? value : 255;
        destination[pixelId].y = value;
        destination[pixelId].z = value;

    // Save last state of the cell to the alpha channel that is not used in rendering.
    destination[pixelId].w = value;
}

void runDisplayLifeKernel(const char* d_lifeData, size_t worldWidth, size_t worldHeight, uchar4* destination,
    int destWidth, int destHeight, int displacementX, int displacementY, int zoom) {

    ushort threadsCount = 128;
    assert((worldWidth * worldHeight) % threadsCount == 0);
    size_t reqBlocksCount = (destWidth * destHeight) / threadsCount;
    assert(reqBlocksCount < 65536);
    ushort blocksCount = (ushort)reqBlocksCount;

    int multisample = std::min(4, (int)std::pow(2, std::max(0, zoom)));

    displayLifeKernel<<<blocksCount, threadsCount>>>(d_lifeData, uint(worldWidth), uint(worldHeight), destination,
        destWidth, destHeight, make_int2(displacementX, displacementY), std::pow(2, zoom),
        multisample);
        cudaDeviceSynchronize();
}

#endif // _CONWAY_CU_


    // for(int i = 0; i < height; i++) {
    //     for(int j = 0; j < width; j++) {
    //         printf("%d ", *d_world_out[i*(width) + j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");