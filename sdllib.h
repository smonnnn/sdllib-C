#include "libs/matrix.h"

#define EULER_NUMBER_F 2.71828182846

typedef struct SDLNet{
	float total_error;
	int layer_count;
	int* layer_sizes;
	Matrix* buffer_1; //Used as temp buffer for bias derivatives.
	Matrix* buffer_2; //Used as temp buffer for weight derivatives.
	Matrix* buffer_3; //Used as temp buffer for error values.
	Matrix* values;
	Matrix* biases;
	Matrix* weights;
	Matrix* output_values;
} SDLNet;

void forward(SDLNet* net, Matrix* input);
void backward(SDLNet* net, Matrix* input, Matrix* target);
void init_network(SDLNet* net, int* layer_sizes, int layer_count);
void delete_network(SDLNet* net);
float sigmoidf_deriv(float n);
float sigmoidf(float n);
void net_print_debug(SDLNet* net, int layer);