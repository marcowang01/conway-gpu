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
# define ITERATIONS 1

# define VERBOSE true 
# define IS_RAND false 

#define BLOCK_SIZE 128
#define BYTES_PER_THREAD 16

// TODO: change back to using ints for better mem access? 
// TODO: change to using share memory
// TODO: compute lookup table and then add later

__global__ void conway_kernel(unsigned* d_world_in, unsigned* d_world_out,  unsigned char* lookup_table,
    int const width, int const height)
{
    int tx = threadIdx.x;  
    int bx = blockIdx.x; 

    int world_size = height * width;
    int tid = bx * blockDim.x + tx;

    // tid is indx of the cell in the world
    // i is the index of every 2 cells, incremented essentially by world size in current setup
    // for (int i = tid * 4 + ; i < world_size; i += blockDim.x * gridDim.x * BYTES_PER_THREAD * 8)
    // {
        // ** Fetch 2 bytes and the rows above and below  as data0,1,2**
        
        int i = tid * BYTES_PER_THREAD / 4;
        if ( (tid / width)  % 2 == 1)
        {
            i = (i + width) % world_size ;
        }

        uint x = (i + width - 1) % width; // evaluates x - 1 first 
        uint y = (i / width) * width; // y: y offest of the cell
        uint yUp = (y + world_size - width) % world_size; // yUp: y offset of the cell above
        uint yDown = (y + width) % world_size; // yDown: y offset of the cell below
        uint yDownDown = (y + 2 * width) % world_size; // yDown: y offset of the cell below
        // 3 integers to hold the 3 rows
        // load in the first byte
        uint64_t data0 = (uint64_t) d_world_in[yUp + x]  << 32;
        uint64_t data1 = (uint64_t) d_world_in[y + x]  << 32;
        uint64_t data2 = (uint64_t) d_world_in[yDown + x]  << 32;
        uint64_t data3 = (uint64_t) d_world_in[yDownDown + x]  << 32;

        x = (x + 1) % width; // load in first uint
        data0 |= d_world_in[yUp + x];
        data1 |= d_world_in[y + x];
        data2 |= d_world_in[yDown + x];
        data3 |= d_world_in[yDownDown + x];

        for (uint j = 0; j < BYTES_PER_THREAD / 4; j++)
        {
            uint currentState = x; // current cell
            x = (x + 1) % width; // load in the 3rd, 4th, 5th, .... byte
            uint newData0 = d_world_in[yUp + x];
            uint newData1 = d_world_in[y + x];
            uint newData2 = d_world_in[yDown + x];
            uint newData3 = d_world_in[yDownDown + x];

            // load in the left most byte of the new int
            data0 = (data0 << 8) | (newData0 >> 24); // the cell to the right and up
            data1 = (data1 << 8) | (newData1 >> 24);// the cell to the right and down
            data2 = (data2 << 8) | (newData2 >> 24); // the cell to the right and down
            data3 = (data3 << 8) | (newData3 >> 24); // the cell to the right and down

            // get the updated state of the cells in the first uint loaded in
            // the bytes we need are at: A: 8, B: 16, C: 24, D: 32
            uint highA = (((data0 & 0x1F800000000) >> 17) | ((data1 & 0x1F800000000) >> 23) | ((data2 & 0x1F800000000) >> 29) | ((data3 & 0x1F800000000) >> 35));
            uint lowA = (((data0 & 0x1F80000000) >> 13) | ((data1 & 0x1F80000000) >> 19) | ((data2 & 0x1F80000000) >> 25) | ((data3 & 0x1F80000000) >> 31));
            uint highB = (((data0 & 0x1F8000000) >> 9) | ((data1 & 0x1F8000000) >> 15) | ((data2 & 0x1F8000000) >> 21) | ((data3 & 0x1F8000000) >> 27));
            uint lowB = (((data0 & 0x1F800000) >> 5) | ((data1 & 0x1F800000) >> 11) | ((data2 & 0x1F800000) >> 17) | ((data3 & 0x1F800000) >> 23));
            uint highC = (((data0 & 0x1F8000) >> 1) | ((data1 & 0x1F80000) >> 7) | ((data2 & 0x1F80000) >> 13) | ((data3 & 0x1F80000) >> 19));
            uint lowC = (((data0 & 0x1F8000) << 3) | ((data1 & 0x1F8000) >> 3) | ((data2 & 0x1F8000) >> 9) | ((data3 & 0x1F8000) >> 15));
            uint highD = (((data0 & 0x1F800) << 7) | ((data1 & 0x1F800) << 1) | ((data2 & 0x1F800) >> 5) | ((data3 & 0x1F800) >> 11));
            uint lowD = (((data0 & 0x1F80) << 11) | ((data1 & 0x1F80) << 5) | ((data2 & 0x1F80) >> 1) | ((data3 & 0x1F80) >> 7));

            uint temp = (data0 & 0x1F800000000 >> 23);

            // if (tid == 0 && j == 0) {
            //     printf("\niter: %d, x = %d, y = %d, yUp = %d, yDown = %d\n", j, x, y, yUp, yDown);
            //     printf("printing data 0 1 2 \n");
            //     for(int i = 63; i >= 0; i--) {
            //         if (i % 8 == 7)
            //             printf(" ");
            //         printf("%llu", (data0 >> i) & 1);
            //     }
            //     printf("\n");
            //     for(int i = 63; i >= 0; i--) {
            //         if (i % 8 == 7)
            //             printf(" ");
            //         printf("%llu", (data1 >> i) & 1);
            //     }
            //     printf("\n");
            //     for(int i = 63; i >= 0; i--) {
            //         if (i % 8 == 7)
            //             printf(" ");
            //         printf("%llu", (data2 >> i) & 1);
            //     }
            //     printf("\n");
            //     for(int i = 63; i >= 0; i--) {
            //         if (i % 8 == 7)
            //             printf(" ");
            //         printf("%llu", (data3 >> i) & 1);
            //     }
            //     printf("\n");
            //     // for(int i = 63; i >= 0; i--) {
            //     //     if (i % 8 == 7)
            //     //         printf(" ");
            //     //     printf("%llu", (((data1 & 0x1F800000000) >> 29) >> i) & 1);
            //     // }
            //     // printf("\n");
            //     // for(int i = 63; i >= 0; i--) {
            //     //     if (i % 8 == 7)
            //     //         printf(" ");
            //     //     printf("%llu", (((data2 & 0x1F800000000) >> 35) >> i) & 1);
            //     // }
            //     printf("\n");
            //     printf("printing A high \n");
            //     for(int i = 23; i >= 0; i--) {
            //         if (i % 6 == 5)
            //             printf("\n");
            //         printf("%d", (highA >> i) & 1);
            //     }
            //     printf("\n");
            //     printf("printing A low \n");
            //     for(int i = 23; i >= 0; i--) {
            //         if (i % 6 == 5)
            //             printf("\n");
            //         printf("%d", (lowA >> i) & 1);
            //     }
            //     printf("\n");
            //     printf("printing A high outcome \n");
            //     for(int i = 8; i >= 0; i--) {
            //         if (i % 8 == 7)
            //             printf(" ");
            //         printf("%d", (((lookup_table[highA])) >> i) & 1);
            //         // printf("%d", ((lookup_table[highA]) >> i) & 1);
            //     }
            //     printf("\n");
            //     printf("printing A low outcome \n");
            //     for(int i = 8; i >= 0; i--) {
            //         if (i % 8 == 7)
            //             printf(" ");
            //         printf("%d", (((lookup_table[lowA])) >> i) & 1);
            //         // printf("%d", ((lookup_table[highA]) >> i) & 1);
            //     }
            //     printf("\n");
            // }

            // get the updated state of the cells in the second uint loaded in
            uint highARes = lookup_table[highA];
            uint lowARes = lookup_table[lowA];
            uint highBRes = lookup_table[highB];
            uint lowBRes = lookup_table[lowB];
            uint highCRes = lookup_table[highC];
            uint lowCRes = lookup_table[lowC];
            uint highDRes = lookup_table[highD];
            uint lowDRes = lookup_table[lowD];


            d_world_out[currentState + yDown] = (
                ((highARes >> 4) << 28) | ((lowARes >> 4) << 24) |
                ((highBRes >> 4) << 20) | ((lowBRes >> 4) << 16) |
                ((highCRes >> 4) << 12) | ((lowCRes >> 4) << 8) |
                ((highDRes >> 4) << 4) | (lowDRes >> 4)
            );

            d_world_out[currentState + y] = (
                ((highARes & 0xF) << 28) | ((lowARes & 0xF) << 24) |
                ((highBRes & 0xF) << 20) | ((lowBRes & 0xF) << 16) |
                ((highCRes & 0xF) << 12) | ((lowCRes & 0xF) << 8) |
                ((highDRes & 0xF) << 4) | (lowDRes & 0xF)
            );

            data0 = data0 << 24 | newData0;
            data1 = data1 << 24 | newData1;
            data2 = data2 << 24 | newData2;
            data3 = data3 << 24 | newData3;
        // }



        // // load in the second byte
        // x = (x + 1) % width; // increment x to the next cell
        // uint data0Original = d_world_in[yUp + x]
        // uint data1Original = d_world_in[y + x]
        // uint data2Original = d_world_in[yDown + x]

        // data0 |= data0Original >> 24; // the cell to the right and up
        // data1 |= data1Original >> 24;// the cell to the right and down
        // data2 |= data2Original >> 24; // the cell to the right and down

        // // loop per uints per thread
        // for (uint j = 0; j < BYTES_PER_THREAD / 4; j++)
        // {
        //     uint currentState = x; // current cell


        //     if (j % 4  == 2) {
        //         x = (x + 1) % width; // load in the 3rd, 4th, 5th, .... byte
        //         data0Original = d_world_in[yUp + x]
        //         data1Original = d_world_in[y + x]
        //         data2Original = d_world_in[yDown + x]
        //     }

        //     data0 = (data0 << 8) | ((data0Original >> 16) & 0xFF); // the cell to the right and up
        //     data1 = (data1 << 8) | ((data1Original >> 16) & 0xFF); // the cell to the right and down
        //     data2 = (data2 << 8) | ((data2Original >> 16) & 0xFF); // the cell to the right and down

        //     x = (x + 1) % width; // load in the 3rd, 4th, 5th, .... byte
        //     data0 = (data0 << 8) | (uint) d_world_in[yUp + x]; // the cell to the right and up
        //     data1 = (data1 << 8) | (uint) d_world_in[y + x]; // the cell to the right and down
        //     data2 = (data2 << 8) | (uint) d_world_in[yDown + x]; // the cell to the right and down
            
            
        //     // encodes 6 * 3 block into one 18 bit number to pass in as a key to the lookup table
        //     uint HighFourBitStates = ((data0 & 0x1F800) << 1) | ((data1 & 0x1F800) >> 5) | ((data2 & 0x1F800) >> 11);
		// 	uint LowFourBitStates = ((data0 & 0x1F80) << 5) | ((data1 & 0x1F80) >> 1) | ((data2 & 0x1F80) >> 7);

        //     // if (tid == 9) {
        //     //     printf("\nx = %d, y = %d, yUp = %d, yDown = %d\n", x, y, yUp, yDown);
        //     //     printf("printing data 0 1 2 \n");
        //     //     for(int i = 31; i >= 0; i--) {
        //     //         if (i % 8 == 7)
        //     //             printf(" ");
        //     //         printf("%d", (data0 >> i) & 1);
        //     //     }
        //     //     printf("\n");
        //     //     for(int i = 31; i >= 0; i--) {
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
        //     // }

        //     // evaluates entire row of 8 cells
        //     d_world_out[currentState + y] = (lookup_table[HighFourBitStates] << 4) | lookup_table[LowFourBitStates];

        //     // if (tid == 9) {
        //     //     printf("printing high \n");
        //     //     for(int i = 31; i >= 0; i--) {
        //     //         if (i % 8 == 7)
        //     //             printf(" ");
        //     //         printf("%d", (HighFourBitStates >> i) & 1);
        //     //     }
        //     //     printf("\n");
        //     //     printf("printing low \n");
        //     //     for(int i = 31; i >= 0; i--) {
        //     //         if (i % 8 == 7)
        //     //             printf(" ");
        //     //         printf("%d", (LowFourBitStates >> i) & 1);
        //     //     }
        //     //     printf("\n");
        //     //     printf("printing result \n");
        //     //     for(int i = 31; i >= 0; i--) {
        //     //         if (i % 8 == 7)
        //     //             printf(" ");
        //     //         printf("%d", (d_world_out[currentState + y] >> i) & 1);
        //     //     }
        //     //     printf("\n");
        //     // }

        // }
    }
} 

void runConwayKernel(unsigned ** d_world_in, unsigned ** d_world_out, unsigned char* lookup_table,
    const int width, const int height, int iterations)
{   
    // TODO: handle case when things are not divisible by 8
    // may need to pad the matrix with the otherside of the matrix
    // assert(((height * width) / 8 / BYTES_PER_THREAD) % BLOCK_SIZE == 0);

    // each thread will process BYTES_PER_THREAD * 8 cells
    // each block will process contiguous BYTES_PER_THREAD * 8 * BLOCK_SIZE cells

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