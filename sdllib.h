#include "libs/matrix.h"

#define EULER_NUMBER_F 2.71828182846

typedef struct SDLNet{
	int layer_count;
	int* layer_sizes;
	Matrix* buffer_1;
	Matrix* buffer_2;
	Matrix* values;
	Matrix* biases;
	Matrix* weights;
	Matrix* output_values;
} SDLNet;

void forward(SDLNet* net, Matrix* input);
void backward(SDLNet* net, Matrix* input, Matrix* goal);
void init_network(SDLNet* net, int* layer_sizes, int layer_count);
float sigmoidf_deriv(float n);
float sigmoidf(float n);