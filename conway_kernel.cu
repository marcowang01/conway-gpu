#ifndef _CONWAY_CU_
#define _CONWAY_CU_

#include <assert.h>

////////////////////////////////////////////////////////////////////////////////
extern "C"
void print_matrix(unsigned *u, int h, int w);
////////////////////////////////////////////////////////////////////////////////

#define BLOCK_SIZE 128

// TODO: change to using bytes instead of ints

__global__ void conway_kernel(unsigned* d_world_in, unsigned* d_world_out, int height, int width)
{
    int tx = threadIdx.x; 
    int bx = blockIdx.x; 

    int world_size = height * width;
    int tid = bx * blockDim.x + tx;

    for (uint i = tid; i < world_size; i += blockDim.x * gridDim.x)
    {
        uint x = i % width;
        uint yAbs = i - x;
        uint xLeft = (x + width - 1) % width;
        uint xRight = (x + 1) % width;
        uint yAbsUp = (yAbs + world_size - width) % world_size;
        uint yAbsDown = (yAbs + width) % world_size;


        uint n = d_world_in[yAbsUp + xLeft] + d_world_in[yAbsUp + x] + d_world_in[yAbsUp + xRight] +
            d_world_in[yAbs + xLeft] + d_world_in[yAbs + xRight] +
            d_world_in[yAbsDown + xLeft] + d_world_in[yAbsDown + x] + d_world_in[yAbsDown + xRight];
        
        d_world_out[i] = (n == 3 || (n == 2 && d_world_in[i]));
    }
} 

void runConwayKernel(unsigned** d_world_in, unsigned** d_world_out, int height, int width, int iterations)
{
    // assert((height * width) % BLOCK_SIZE == 0);
    size_t numBlocks = (height * width) / BLOCK_SIZE;
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(numBlocks);

    for (int i = 0; i < iterations; i++)
    {
        conway_kernel<<<dimGrid, dimBlock>>>(*d_world_in, *d_world_out, height, width);
        // cudaDeviceSynchronize();
        // swap pointers
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