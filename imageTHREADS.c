#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <string.h>
#include "image.h"
#include <pthread.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

//An array of kernel matrices to be used for image convolution.  
//The indexes of these match the enumeration from the header file. ie. algorithms[BLUR] returns the kernel corresponding to a box blur.
Matrix algorithms[]={
    {{0,-1,0},{-1,4,-1},{0,-1,0}},
    {{0,-1,0},{-1,5,-1},{0,-1,0}},
    {{1/9.0,1/9.0,1/9.0},{1/9.0,1/9.0,1/9.0},{1/9.0,1/9.0,1/9.0}},
    {{1.0/16,1.0/8,1.0/16},{1.0/8,1.0/4,1.0/8},{1.0/16,1.0/8,1.0/16}},
    {{-2,-1,0},{-1,1,1},{0,1,2}},
    {{0,0,0},{0,1,0},{0,0,0}}
};


//getPixelValue - Computes the value of a specific pixel on a specific channel using the selected convolution kernel
//Paramters: srcImage:  An Image struct populated with the image being convoluted
//           x: The x coordinate of the pixel
//          y: The y coordinate of the pixel
//          bit: The color channel being manipulated
//          algorithm: The 3x3 kernel matrix to use for the convolution
//Returns: The new value for this x,y pixel and bit channel
uint8_t getPixelValue(Image* srcImage,int x,int y,int bit,Matrix algorithm){
    int px,mx,py,my,i,span;
    span=srcImage->width*srcImage->bpp;
    // for the edge pixes, just reuse the edge pixel
    px=x+1; py=y+1; mx=x-1; my=y-1;
    if (mx<0) mx=0;
    if (my<0) my=0;
    if (px>=srcImage->width) px=srcImage->width-1;
    if (py>=srcImage->height) py=srcImage->height-1;
    uint8_t result=
        algorithm[0][0]*srcImage->data[Index(mx,my,srcImage->width,bit,srcImage->bpp)]+
        algorithm[0][1]*srcImage->data[Index(x,my,srcImage->width,bit,srcImage->bpp)]+
        algorithm[0][2]*srcImage->data[Index(px,my,srcImage->width,bit,srcImage->bpp)]+
        algorithm[1][0]*srcImage->data[Index(mx,y,srcImage->width,bit,srcImage->bpp)]+
        algorithm[1][1]*srcImage->data[Index(x,y,srcImage->width,bit,srcImage->bpp)]+
        algorithm[1][2]*srcImage->data[Index(px,y,srcImage->width,bit,srcImage->bpp)]+
        algorithm[2][0]*srcImage->data[Index(mx,py,srcImage->width,bit,srcImage->bpp)]+
        algorithm[2][1]*srcImage->data[Index(x,py,srcImage->width,bit,srcImage->bpp)]+
        algorithm[2][2]*srcImage->data[Index(px,py,srcImage->width,bit,srcImage->bpp)];
    return result;
}

//so each thread knows what part of image it should work on, one struct per thread. helps avoid race conditions
typedef struct {
	Image* src;
	Image* dest;
	int y_start;
	int y_end;
	int kernel_index;
} Work;

static void* worker(void* arg) {  //each thread runs this to know to only loop through its own rows
	Work* w = (Work*)arg;
	Image* src = w->src;
	Image* dest = w->dest;
	int kernel = w->kernel_index;

	for (int row = w->y_start; row < w->y_end; row++){
        	for (int x = 0; x < src->width; x++){
            		for (int bit = 0; bit < src->bpp; bit++){
                		dest->data[Index(x, row, src->width, bit, src->bpp)] =
                    		getPixelValue(src, x, row, bit, algorithms[kernel]);
            		}	
        	}
    	}	
    	return NULL;
}	

// altered to allocate arrays and evenly divide rows among threads.	
void convolute_thread(Image* src, Image* dst, int kernel_index, int num_threads){
    if (num_threads < 1) num_threads = 1;
    if (num_threads > src->height) num_threads = src->height;

    pthread_t* threadID = (pthread_t*)malloc(sizeof(pthread_t)*num_threads);
    Work* jobs = (Work*)malloc(sizeof(Work)*num_threads);

    int height = src->height;
    for (int t = 0; t < num_threads; t++){
        int y0 = (t * height) / num_threads;
        int y1 = ((t + 1) * height) / num_threads;

        jobs[t].src = src;
        jobs[t].dest = dst;
        jobs[t].y_start = y0;
        jobs[t].y_end   = y1;
        jobs[t].kernel_index = kernel_index;

        pthread_create(&threadID[t], NULL, worker, &jobs[t]);
    }
    for (int t = 0; t < num_threads; t++){
        pthread_join(threadID[t], NULL);
    }
    free(threadID);
    free(jobs);
}


//Usage: Prints usage information for the program
//Returns: -1
int Usage(){
    printf("Usage: image <filename> <type>\n\twhere type is one of (edge,sharpen,blur,gauss,emboss,identity)\n");
    return -1;
}

//GetKernelType: Converts the string name of a convolution into a value from the KernelTypes enumeration
//Parameters: type: A string representation of the type
//Returns: an appropriate entry from the KernelTypes enumeration, defaults to IDENTITY, which does nothing but copy the image.
enum KernelTypes GetKernelType(char* type){
    if (!strcmp(type,"edge")) return EDGE;
    else if (!strcmp(type,"sharpen")) return SHARPEN;
    else if (!strcmp(type,"blur")) return BLUR;
    else if (!strcmp(type,"gauss")) return GAUSE_BLUR;
    else if (!strcmp(type,"emboss")) return EMBOSS;
    else return IDENTITY;
}

//main:
//argv is expected to take 2 arguments.  First is the source file name (can be jpg, png, bmp, tga).  Second is the lower case name of the algorithm.
int main(int argc, char** argv){
    if (argc < 3 || argc > 4) return Usage();

    char* fileName = argv[1];
    enum KernelTypes ktype = GetKernelType(argv[2]);
    int threads = (argc == 4) ? atoi(argv[3]) : 4;
    if (threads < 1) threads = 1;

    if (!strcmp(argv[1],"pic4.jpg") && !strcmp(argv[2],"gauss")){
        printf("You have applied a gaussian filter to Gauss which has caused a tear in the time-space continum.\n");
    }

    Image src, dst;
    src.data = stbi_load(fileName, &src.width, &src.height, &src.bpp, 0);
    if (!src.data){
        printf("Error loading file %s.\n", fileName);
        return -1;
    }

    dst.width = src.width;
    dst.height = src.height;
    dst.bpp = src.bpp;
    size_t nbytes = (size_t)dst.width * dst.height * dst.bpp;
    dst.data = (uint8_t*)malloc(nbytes);
    if (!dst.data){
        printf("Failed to allocate destination image.\n");
        stbi_image_free(src.data);
        return -1;
    }

    long t1 = time(NULL);
    convolute_thread(&src, &dst, (int)ktype, threads);
    long t2 = time(NULL);

    stbi_write_png("output.png", dst.width, dst.height, dst.bpp, dst.data, dst.width * dst.bpp);

    stbi_image_free(src.data);
    free(dst.data);

    printf("Took %ld seconds\n", t2 - t1);
    return 0;
}
