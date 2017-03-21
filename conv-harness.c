 /* Test and timing harness program for developing a multichannel
   multikernel convolution (as used in deep learning networks)

   Note there are some simplifications around this implementation,
   in particular with respect to computing the convolution at edge
   pixels of the image.

   Author: David Gregg
   Date:   February 2017

   Version 1.3 : Fixed which loop variables were being incremented
				 in write_out();
				 Fixed dimensions of output and control_output 
				 matrices in main function

   Version 1.2 : Changed distribution of test data to (hopefully) 
				 eliminate random walk of floating point error;
				 Also introduced checks to restrict kernel-order to
				 a small set of values

   Version 1.1 : Fixed bug in code to create 4d matrix
*/

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>
#include <omp.h>
#include <math.h>

/* the following two definitions of DEBUGGING control whether or not
   debugging information is written out. To put the program into
   debugging mode, uncomment the following line: */
//#define DEBUGGING(_x) _x
/* to stop the printing of debugging information, use the following line: */
#define DEBUGGING(_x)


/* write 3d matrix to stdout */
void write_out(float *** a, int dim0, int dim1, int dim2)
{
  int i, j, k;

  for ( i = 0; i < dim0; i++ ) {
	printf("Outer dimension number %d\n", i);
	for ( j = 0; j < dim1; j++ ) {
	  for ( k = 0; k < dim2 - 1; k++ ) {
		printf("%f, ", a[i][j][k]);
	  }
	  // print end of line
	  printf("%f\n", a[i][j][dim2-1]);
	}
  }
}


/* create new empty 4d matrix */
float **** new_empty_4d_matrix(int dim0, int dim1, int dim2, int dim3)
{
  float **** result = malloc(dim0 * sizeof(float***));
  float *** mat1 = malloc(dim0 * dim1 * sizeof(float**));
  float ** mat2 = malloc(dim0 * dim1 * dim2 * sizeof(float*));
  float * mat3 = malloc(dim0 * dim1 * dim2 *dim3 * sizeof(float));
  int i, j, k;

  
  for ( i = 0; i < dim0; i++ ) {
	result[i] = &(mat1[i*dim1]);
	for ( j = 0; j < dim1; j++ ) {
	  result[i][j] = &(mat2[i*dim1*dim2 + j*dim2]);
	  for ( k = 0; k < dim2; k++ ) {
		result[i][j][k] = &(mat3[i*dim1*dim2*dim3+j*dim2*dim3+k*dim3]);
	  }
	}
  }

  return result;
}

/* create new empty 3d matrix */
float *** new_empty_3d_matrix(int dim0, int dim1, int dim2)
{
  float **** mat4d;
  float *** mat3d;

  // create a 4d matrix with single first dimension
  mat4d = new_empty_4d_matrix(1, dim0, dim1, dim2);
  // now throw away out first dimension
  mat3d = mat4d[0];
  free(mat4d);
  return mat3d;
}

/* take a copy of the matrix and return in a newly allocated matrix */
float **** copy_4d_matrix(float **** source_matrix, int dim0,
							int dim1, int dim2, int dim3)
{
  int i, j, k, l;
  float **** result = new_empty_4d_matrix(dim0, dim1, dim2, dim3);

  for ( i = 0; i < dim0; i++ ) {
	for ( j = 0; j < dim1; j++ ) {
	  for ( k = 0; k < dim2; k++ ) {
		for ( l = 0; l < dim3; l++ ) {
		  result[i][j][k][l] = source_matrix[i][j][k][l];
		}
	  }
	}
  }
  return result;
}

/* create a matrix and fill it with random numbers */
float **** gen_random_4d_matrix(int dim0, int dim1, int dim2, int dim3)
{
float **** result;
int i, j, k, l;
struct timeval seedtime;
  int seed;

  result = new_empty_4d_matrix(dim0, dim1, dim2, dim3);

  /* use the microsecond part of the current time as a pseudorandom seed */
  gettimeofday(&seedtime, NULL);
  seed = seedtime.tv_usec;
  srandom(seed);

  /* fill the matrix with random numbers */
  const int range = 1 << 16; // 2^16
  const int bias = 1 << 12; // 2^12
  float offset = 4.0;
  for ( i = 0; i < dim0; i++ ) {
	for ( j = 0; j < dim1; j++ ) {
	  for ( k = 0; k < dim2; k++ ) {
		for ( l = 0; l < dim3; l++ ) {
		  // generate uniform random integer with mean of zero
		  long long rand = random();
		  // now cut down the range and bias the mean to reduce
		  // the likelihood of large floating point round-off errors
		  int reduced_range = (rand % range);
		  float num = (((float) reduced_range) / ((float) bias))+offset;
		  result[i][j][k][l] = num;
		}
	  }
	}
  }

  return result;
}

/* create a matrix and fill it with random numbers */
float *** gen_random_3d_matrix(int dim0, int dim1, int dim2)
{
  float **** mat4d;
  float *** mat3d;

  // create a 4d matrix with single first dimension
  mat4d = gen_random_4d_matrix(1, dim0, dim1, dim2);
  // now throw away out first dimension
  mat3d = mat4d[0];
  free(mat4d);
  return mat3d;
}

/* check the sum of absolute differences is within reasonable epsilon */
void check_result(float *** result, float *** control,
				  int dim0, int dim1, int dim2)
{
  int i, j, k;
  double sum_abs_diff = 0.0;
  const double EPSILON = 0.0625;

  //printf("SAD\n");
  
  for ( i = 0; i < dim0; i++ ) {
	for ( j = 0; j < dim1; j++ ) {
	  for ( k = 0; k < dim2; k++ ) {
		double diff = fabs(control[i][j][k] - result[i][j][k]);
		assert( diff >= 0.0 );
		sum_abs_diff = sum_abs_diff + diff;
	  }
	}
  }

  if ( sum_abs_diff > EPSILON ) {
	fprintf(stderr, "WARNING: sum of absolute differences (%f) > EPSILON (%f)\n",
			sum_abs_diff, EPSILON);
  }
  else {
	printf("COMMENT: sum of absolute differences (%f)  within acceptable range (%f)\n", sum_abs_diff, EPSILON);
  }
}

/* the slow but correct version of matmul written by David */
void multichannel_conv(float *** image, float **** kernels, float *** output,
					   int width, int height, int nchannels, int nkernels,
					   int kernel_order)
{
  int h, w, x, y, c, m;

  for ( m = 0; m < nkernels; m++ ) {
    for ( w = 0; w < width; w++ ) {
      for ( h = 0; h < height; h++ ) {
        float sum = 0.0;
        for ( c = 0; c < nchannels; c++ ) {
          for ( x = 0; x < kernel_order; x++) {
            for ( y = 0; y < kernel_order; y++ ) {
              sum += image[w+x][h+y][c] * kernels[m][c][x][y];
            }
          }
          output[m][w][h] = sum;
        }
      }
    }
  }
}
/*Our version that is not parallelized*/
void team_conv_not_parallel(float *** image, float **** kernels, float *** output,
			   int width, int height, int nchannels, int nkernels,
			   int kernel_order)
{


int h, w,x, y, c, m;
float sum;
float sum2;

  for ( m = 0; m < nkernels; m++ ) {
	for ( w = 0; w < width; w++ ) {
	  for ( h = 0; h < height; h+=2 ) {
		//do the thing in the slides where you reduce number of memory accesses
		//gives diminishing returns though
		 sum = 0.0;
	     sum2 = 0.0;

		for ( c = 0; c < nchannels; c++ ) {
		  for ( x = 0; x < kernel_order; x++) {
			for ( y = 0; y < kernel_order; y++ ) {
			  sum += image[w+x][h+y][c] * kernels[m][c][x][y];
			  sum2 += image[w+x][h+1+y][c] * kernels[m][c][x][y];

			}
		  }
		  output[m][w][h] = sum;
		  output[m][w][h+1]=sum2;

		}
	  }
	}
  }


}

/* the fast version of matmul written by the team */
void team_conv(float *** image, float **** kernels, float *** output,
			   int width, int height, int nchannels, int nkernels,
			   int kernel_order)
{
/*Our version*/

if(width < 50 || nchannels || 4 || nkernels || 4 || kernel_order < 4) team_conv_not_parallel(image, kernels, output, width, height, nchannels, nkernels, kernel_order);
else{

int h, w,x, y, c, m;
float sum;
float sum2;

#pragma omp parallel for private(h, w, x, y, c, m, sum, sum2)
  for ( m = 0; m < nkernels; m++ ) {
	for ( w = 0; w < width; w++ ) {
	  for ( h = 0; h < height; h+=2 ) {
		//do the thing in the slides where you reduce number of memory accesses
		//gives diminishing returns though
		 sum = 0.0;
	     sum2 = 0.0;

		for ( c = 0; c < nchannels; c++ ) {
		  for ( x = 0; x < kernel_order; x++) {
			
			for ( y = 0; y < kernel_order; y++ ) {
					
					  sum += image[w+x][h+y][c] * kernels[m][c][x][y];
					  sum2 += image[w+x][h+1+y][c] * kernels[m][c][x][y];
			}
			
		  }
		  output[m][w][h] = sum;
		  output[m][w][h+1]=sum2;

		}
	  }
	}
  
  }

}
}


int main(int argc, char ** argv)
{
  //float image[W][H][C];
  //float kernels[M][C][K][K];
  //float output[M][W][H];
  
	float *** image, **** kernels, *** output;
	float *** control_output;
	long long mul_time;
	int width, height, kernel_order, nchannels, nkernels;
 	struct timeval start_time;
	struct timeval stop_time;

	if ( argc != 6 ) {
		fprintf(stderr, "Usage: conv-harness <image_width> <image_height> <kernel_order> <number of channels> <number of kernels>\n");
		exit(1);
	}
	else {
		width = atoi(argv[1]);
		height = atoi(argv[2]);
		kernel_order = atoi(argv[3]);
		nchannels = atoi(argv[4]);
		nkernels = atoi(argv[5]);
	}
	switch ( kernel_order ) {
		case 1:
		case 3:
		case 5:
		case 7: break;
		default:
			fprintf(stderr, "FATAL: kernel_order must be 1, 3, 5 or 7, not %d\n",
				kernel_order);
		exit(1);
	}

	/* allocate the matrices */
	image = gen_random_3d_matrix(width+kernel_order, height + kernel_order,
							   nchannels);
	kernels = gen_random_4d_matrix(nkernels, nchannels, kernel_order, kernel_order);
	output = new_empty_3d_matrix(nkernels, width, height);
	control_output = new_empty_3d_matrix(nkernels, width, height);

  //DEBUGGING(write_out(A, a_dim1, a_dim2));

	/* record starting time of David's code*/
	gettimeofday(&start_time, NULL);
	/* use a simple multichannel convolution routine to produce control result */
	multichannel_conv(image, kernels, control_output, width,
					height, nchannels, nkernels, kernel_order);
	/* record finishing time */
	gettimeofday(&stop_time, NULL);
	mul_time = (stop_time.tv_sec - start_time.tv_sec) * 1000000L +
	(stop_time.tv_usec - start_time.tv_usec);
	printf("David's conv time: %lld microseconds\n", mul_time);

	long long davidsTime = mul_time;

	/* record starting time of team's code*/
	gettimeofday(&start_time, NULL);

	/* perform student team's multichannel convolution */
	team_conv(image, kernels, output, width,
					height, nchannels, nkernels, kernel_order);

	/* record finishing time */
	gettimeofday(&stop_time, NULL);
	mul_time = (stop_time.tv_sec - start_time.tv_sec) * 1000000L +
	(stop_time.tv_usec - start_time.tv_usec);
	printf("Team conv time: %lld microseconds\n", mul_time);

	long long teamsTime = mul_time;
	
	long long factor = davidsTime / teamsTime;
	
	if(factor!=0) printf("Your implementation is between %lld and %lld times faster\n", factor, factor+1);

	DEBUGGING(write_out(output, nkernels, width, height));

	/* now check that the team's multichannel convolution routine
	 gives the same answer as the known working version */
	check_result(output, control_output, nkernels, width, height);

  return 0;
}
