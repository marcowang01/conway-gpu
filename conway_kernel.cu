#ifndef _CONWAY_CU_
#define _CONWAY_CU_

#include <assert.h>

////////////////////////////////////////////////////////////////////////////////
extern "C"
void print_matrix(unsigned *u, int h, int w);
////////////////////////////////////////////////////////////////////////////////

#define BLOCK_X 32
#define BLOCK_Y 32

// use unsigned char instead of ints

// __global__ void simpleLifeKernel(const ubyte* lifeData, uint worldWidth,
//     uint worldHeight, ubyte* resultLifeData) {
//   uint worldSize = worldWidth * worldHeight;
 
//   for (uint cellId = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
//       cellId < worldSize;
//       cellId += blockDim.x * gridDim.x) {
//     uint x = cellId % worldWidth;
//     uint yAbs = cellId - x;
//     uint xLeft = (x + worldWidth - 1) % worldWidth;
//     uint xRight = (x + 1) % worldWidth;
//     uint yAbsUp = (yAbs + worldSize - worldWidth) % worldSize;
//     uint yAbsDown = (yAbs + worldWidth) % worldSize;
 
//     uint aliveCells = lifeData[xLeft + yAbsUp] + lifeData[x + yAbsUp]
//       + lifeData[xRight + yAbsUp] + lifeData[xLeft + yAbs] + lifeData[xRight + yAbs]
//       + lifeData[xLeft + yAbsDown] + lifeData[x + yAbsDown] + lifeData[xRight + yAbsDown];
 
//     resultLifeData[x + yAbs] =
//       aliveCells == 3 || (aliveCells == 2 && lifeData[x + yAbs]) ? 1 : 0;
//   }
// }

// void runSimpleLifeKernel(ubyte*& d_lifeData, ubyte*& d_lifeDataBuffer, size_t worldWidth,
//     size_t worldHeight, size_t iterationsCount, ushort threadsCount) {
//   assert((worldWidth * worldHeight) % threadsCount == 0);
//   size_t reqBlocksCount = (worldWidth * worldHeight) / threadsCount;
//   ushort blocksCount = (ushort)std::min((size_t)32768, reqBlocksCount);
 
//   for (size_t i = 0; i < iterationsCount; ++i) {
//     simpleLifeKernel<<<blocksCount, threadsCount>>>(d_lifeData, worldWidth,
//       worldHeight, d_lifeDataBuffer);
//     std::swap(d_lifeData, d_lifeDataBuffer);
//   }
// }


__global__ void conway_kernel(unsigned* d_world_in, unsigned* d_world_out, int height, int width)
{
    int tx = threadIdx.x; int ty = threadIdx.y;
    int bx = blockIdx.x; int by = blockIdx.y;

    int index_x = bx * (blockDim.x - 2) + tx;
    int index_y = by * (blockDim.y - 2) + ty;
    int sh_x = tx;
    int sh_y = ty;
    int middle_square_pos = index_y * width + index_x;
    // dimensions should blockdim.y * blockdim.x
    __shared__ int shared_world[BLOCK_Y][BLOCK_X];

    if((index_x) < (width) && index_y < (height)) {
        shared_world[ty][tx] = d_world_in[middle_square_pos];
    }

    __syncthreads();

    if((index_x) < (width-1) && index_y < (height-1)) {
        if((sh_x > 0) && (sh_x < (blockDim.x - 1)) && (sh_y > 0) && (sh_y < (blockDim.y - 1))) {
            unsigned n = shared_world[sh_y-1][sh_x-1] + shared_world[sh_y-1][sh_x] 
                + shared_world[sh_y-1][sh_x+1] + shared_world[sh_y][sh_x-1] 
                + shared_world[sh_y][sh_x+1] + shared_world[sh_y+1][sh_x-1] 
                + shared_world[sh_y+1][sh_x] + shared_world[sh_y+1][sh_x+1];
            // d_world_out[middle_square_pos] = (n == 3 || (n == 2 && shared_world[sh_y][sh_x]));
            d_world_out[middle_square_pos] = 9;
        }
        
    }
    
    __syncthreads();
} 

void runConwayKernel(unsigned* d_world_in, unsigned* d_world_out, int height, int width, int iterations)
{
    int grid_height, grid_width;

    grid_height = (height+BLOCK_Y-3) / (BLOCK_Y-2);
    grid_width = (width+BLOCK_X-3) / (BLOCK_X-2);
    
    dim3 dimBlock(BLOCK_Y, BLOCK_X);
    dim3 dimGrid(grid_height, grid_width);

    for (int i = 0; i < iterations; i++)
    {
        conway_kernel<<<dimGrid, dimBlock>>>(d_world_in, d_world_out, height, width);
        cudaDeviceSynchronize();
        // if running for startup then don't swap
        if (iterations == 1)
            return;
        unsigned* temp = d_world_in;
        d_world_in = d_world_out;
        d_world_out = temp;
        // print_matrix(d_world_in, height, width);
    }
}

#endif // _CONWAY_CU_