#ifndef _CONWAY_CU_
#define _CONWAY_CU_

#include <assert.h>

////////////////////////////////////////////////////////////////////////////////
extern "C" 
void printBinary(unsigned n);

extern "C" 
void printMatrix(unsigned *u, int h, int w);
////////////////////////////////////////////////////////////////////////////////

#define BLOCK_SIZE 32
#define BYTES_PER_THREAD 2

// TODO: change back to using ints for better mem access? 
// TODO: change to using share memory
// TODO: compute lookup table and then add later

__global__ void conway_kernel(unsigned char* d_world_in, unsigned char* d_world_out,  unsigned char* lookup_table,
    int const height, int const width)
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

        // if (tid == 4) {
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
        // }

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

void runConwayKernel(unsigned char** d_world_in, unsigned char** d_world_out, unsigned char* lookup_table,
    const int height, const int width, int iterations)
{   
    // TODO: handle case when things are not divisible by 8
    // may need to pad the matrix with the otherside of the matrix
    assert(((height * width) / 8 / BYTES_PER_THREAD) % BLOCK_SIZE == 0);

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
        // FIXME: later on can clamp this to 32768 as max number of blocks
        conway_kernel<<<dimGrid, dimBlock>>>(*d_world_in, *d_world_out, lookup_table, height, width / 8);
        // cudaDeviceSynchronize();
        std::swap(d_world_in, d_world_out);
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