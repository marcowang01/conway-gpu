#ifndef _CONWAY_CU_
#define _CONWAY_CU_

#include <assert.h>

////////////////////////////////////////////////////////////////////////////////
extern "C"
void print_matrix(unsigned *u, int h, int w);
////////////////////////////////////////////////////////////////////////////////

#define BLOCK_SIZE 128

// use unsigned char instead of ints

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

    // int index_x = bx * (blockDim.x - 2) + tx;
    // int index_y = by * (blockDim.y - 2) + ty;
    // int sh_x = tx;
    // int sh_y = ty;
    // int middle_square_pos = index_y * width + index_x;
    // // dimensions should blockdim.y * blockdim.x
    // __shared__ int shared_world[BLOCK_Y][BLOCK_X];

    // if((index_x) < (width) && index_y < (height)) {
    //     shared_world[ty][tx] = d_world_in[middle_square_pos];
    // }

    // __syncthreads();

    // if((index_x) < (width-1) && index_y < (height-1)) {
    //     if((sh_x > 0) && (sh_x < (blockDim.x - 1)) && (sh_y > 0) && (sh_y < (blockDim.y - 1))) {
    //         unsigned n = shared_world[sh_y-1][sh_x-1] + shared_world[sh_y-1][sh_x] 
    //             + shared_world[sh_y-1][sh_x+1] + shared_world[sh_y][sh_x-1] 
    //             + shared_world[sh_y][sh_x+1] + shared_world[sh_y+1][sh_x-1] 
    //             + shared_world[sh_y+1][sh_x] + shared_world[sh_y+1][sh_x+1];
    //         // d_world_out[middle_square_pos] = (n == 3 || (n == 2 && shared_world[sh_y][sh_x]));
    //         d_world_out[middle_square_pos] = 9;
    //     }
        
    // }
    
    // __syncthreads();
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


    // grid_height = (height+BLOCK_Y-3) / (BLOCK_Y-2);
    // grid_width = (width+BLOCK_X-3) / (BLOCK_X-2);
    
    // dim3 dimBlock(BLOCK_Y, BLOCK_X);
    // dim3 dimGrid(grid_height, grid_width);

    // for (int i = 0; i < iterations; i++)
    // {
    //     conway_kernel<<<dimGrid, dimBlock>>>(*d_world_in, *d_world_out, height, width);
    //     cudaDeviceSynchronize();
    //     // if running for startup then don't swap
    //     if (iterations == 1)
    //         return;
    //     unsigned** temp = d_world_in;
    //     d_world_in = d_world_out;
    //     d_world_out = temp;
    // }
}

#endif // _CONWAY_CU_


        // for(int i = 0; i < height; i++) {
        //     for(int j = 0; j < width; j++) {
        //         printf("%d ", *d_world_out[i*(width) + j]);
        //     }
        //     printf("\n");
        // }
        // printf("\n");