#include <math.h>
#include "sdllib.h"
#include "libs/matrix.h"

void init_network(SDLNet* net, int* layer_sizes, int layer_count){
	if(layer_count < 1){
		printf("Layer count cannot be smaller than 1!\n");
		exit(1);
	}

	net->layer_count = layer_count;
	net->layer_sizes = layer_sizes;

	net->values = 	calloc((layer_count * 3) - 2 , sizeof(Matrix));
	net->biases = 	net->values + layer_count;
	net->weights = 	net->biases + layer_count - 1;

	net->values[0] = mat_new(layer_sizes[0], 1);
	Matrix* largest_weights_matrix = net->weights;
	Matrix* largest_values_matrix = net->values;
	for(int i = 1; i < layer_count; i++){
		net->values[i] = 	mat_new(layer_sizes[i], 1);
		net->biases[i-1] = 	mat_new(layer_sizes[i], 1);
		net->weights[i-1] = mat_new(layer_sizes[i-1], layer_sizes[i]);
		if(net->weights[i-1] > largest_weights_matrix->size) {
			largest_weights_matrix = (net->weights + i - 1);
		}
		if(net->values[i] > largest_values_matrix->size) {
			largest_values_matrix = (net->values + i);
		}
	}
	net->output_values = net->values + (layer_count - 1);
	net->buffer_1 = mat_new(largest_values_matrix->width, largest_values_matrix->height);
	net->buffer_2 = mat_new(largest_weights_matrix->width, largest_weights_matrix->height);
}

void forward(SDLNet* net, Matrix* input){
	net->values[0] = *input;
	for(int i = 0; i < net->layer_count - 1; i++){
		mat_mult_matrix(	(net->values + i), 		(net->weights + i), 	(net->values + i + 1));
		mat_add_matrix(		(net->values + i + 1), 	(net->biases + i), 		(net->values + i + 1));
		mat_apply_function(	(net->values + i + 1), 	(net->values + i + 1), 	&sigmoidf);
	}
	printf("Forward pass results:\n");
	mat_print(net->output_values);
}

void backward(SDLNet* net, Matrix* input, Matrix* goal_buffer){
	forward(net, input);

	Matrix buffer_1;
	Matrix buffer_2;

	for(int i = net->layer_count - 1; i > 0; i--){
		buffer_1 = mat_resize_unsafe(net->buffer_1, net->values[i].width, net->values[i].height);
		buffer_2 = mat_resize_unsafe(net->buffer_2, net->weights[i - 1].width, net->weights[i - 1].height);

		//Calculate error, store it in goal_buffer
		mat_subtract_matrix(net->output_values, goal_buffer, goal_buffer);
		//Apply the derivative of the sigmoid function to the values, store it in buffer_1.
		mat_apply_function(net->output_values, buffer_1, &sigmoidf_deriv);
		//Multiply element wise.
		mat_element_wise_mult(&buffer_1, goal_buffer, &buffer_1);
		//buffer_1 now contains the derivative with respect to the bias.
		//Repair goal buffer
		mat_subtract_matrix(net->output_values, goal_buffer, goal_buffer);
		//Apply biases.
		mat_subtract_matrix(net->biases + i - 1, &buffer_1, net->biases + i - 1);
		//Calculate the derivative with respect to the weights. Store it in buffer_2.
		mat_mult_matrix(&buffer_1, net->weights + i - 1, &buffer_2);
		//Apply weights.
		mat_subtract_matrix(net->weights + i - 1, &buffer_2, net->weights + i - 1);
	}
}

float sigmoidf_deriv(float n) {
	return n * (1 - n);
}

float sigmoidf(float n) {
    return (1 / (1 + powl(EULER_NUMBER_F, -n)));
}