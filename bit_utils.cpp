#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

//////////////////////////////////////////////////////////////////////////////////
extern "C"
void printBinary(unsigned n);

extern "C"
void bitPerCellEncode(unsigned *in, unsigned *out, int width, int height);

extern "C"
void bitPerCellDecode(unsigned *in, unsigned *out, int width, int height);

////////////////////////////////////////////////////////////////////////////////
//! Helpers for encoding and decoding unsigned ints to/from 32-bit representation
////////////////////////////////////////////////////////////////////////////////
// convert a 32-bit unsigned int per cell to 32-bit per cell representation
// i.e. [1,0,1,0] -> [bx1010] -> [5] 
void bitPerCellEncode(unsigned *in, unsigned *out, int width, int height)
{
    for (int i = 0; i < width*height/32; i++) {
        out[i] = 0;
    }
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // cell is either 0 or 1

            unsigned int cell = in[y*width+x];
            if (cell == 1) {
                out[(y*width+x) / 32] |= 1 << (31 - (y*width+x) % 32);
            }
        }
    }
}

// convert a 32-bit per cell representation to 32-bit unsigned int per cell
// i.e. [5] -> [bx1010] -> [1,0,1,0]
void bitPerCellDecode(unsigned *in, unsigned *out, int width, int height)
{
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // cell is either 0 or 1
            unsigned int cell = in[(y*width+x) / 32] & (1 << (31 - (y*width+x) % 32));
            if (cell) {
                out[y*width+x] = 1;
            } else {
                out[y*width+x] = 0;
            }
        }
    }
}

void printBinary(unsigned n) {
    for(int i = 31; i >= 0; i--) {
        if (i % 8 == 7)
            printf(" ");
        printf("%d", (n >> i) & 1);
    }
}
