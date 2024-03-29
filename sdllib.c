#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "sdllib.h"

//Used to initialise a new network.
void init_network(SDLNet* net, int* layer_sizes, int layer_count){
	if(layer_count < 1){
		printf("Layer count cannot be smaller than 1!\n");
		exit(1);
	}

	net->total_error = 1.0f;
	net->layer_count = layer_count;
	net->layer_sizes = layer_sizes;

	net->values   = calloc((layer_count * 3) + 1, sizeof(Matrix));
	net->biases   = net->values + layer_count;
	net->weights  = net->biases + layer_count - 1;
	net->buffer_1 = net->weights + layer_count - 1;
	net->buffer_2 = net->buffer_1 + 1;
	net->buffer_3 = net->buffer_1 + 2;

	net->values[0] = mat_new(1, layer_sizes[0]);
	Matrix* largest_weights_matrix  = net->weights;
	Matrix* largest_values_matrix   = net->values;

	for(int i = 1; i < layer_count; i++){
		net->values[i]    = mat_new(1, layer_sizes[i]);
		net->biases[i-1]  = mat_new_random_10(1, layer_sizes[i]);
		net->weights[i-1] = mat_new_random_10(layer_sizes[i-1], layer_sizes[i]);

		if(net->weights[i-1].size > largest_weights_matrix->size) {
			largest_weights_matrix = (net->weights + i - 1);
		}
		if(net->values[i].size > largest_values_matrix->size) {
			largest_values_matrix = (net->values + i);
		}
	}
	net->output_values  = net->values + (layer_count - 1);
	net->buffer_1[0]    = mat_new(largest_values_matrix->width, largest_values_matrix->height);
	net->buffer_2[0]    = mat_new(largest_weights_matrix->width, largest_weights_matrix->height);
	net->buffer_3[0]    = mat_new(largest_values_matrix->width, largest_values_matrix->height);
}

//Used to delete a network after usage.
void delete_network(SDLNet* net){
	for(int i = 0; i < (net->layer_count * 3) + 1; i++){
		mat_delete(net->values + i);
	}
	free(net->values);
}

//Propagate forward through the network.
void forward(SDLNet* net, Matrix* input){
	net->values[0] = *input;
	for(int i = 0; i < net->layer_count - 1; i++){
		mat_mult_matrix((net->weights + i), (net->values + i), (net->values + i + 1));
		mat_add_matrix((net->values + i + 1), (net->biases + i), (net->values + i + 1));
		mat_apply_function((net->values + i + 1), (net->values + i + 1), &sigmoidf);
	}
}

//Propagate backward through the network.
void backward(SDLNet* net, Matrix* input, Matrix* target){
	Matrix *b1, *b2, *b3, *v, *vp, *w, *b;
	b1 = net->buffer_1;
	b2 = net->buffer_2;
	b3 = net->buffer_3;

	forward(net, input);

	//Calculate error, store it.
	mat_resize_unsafe(b3, net->output_values->width, net->output_values->height);
	mat_subtract_matrix(net->output_values, target, b3);

	net->total_error = 0.0f;
	for(int i = 0; i < b3->size; i++){
		net->total_error += b3->data[i] * b3->data[i];
	}
	
	for(int i = net->layer_count - 1; i > 0; i--){
		v = net->values + i;
		vp = net->values + i - 1;
		w = net->weights + i - 1;
		b = net->biases + i - 1;

		mat_resize_unsafe(b1, v->width, v->height);
		mat_resize_unsafe(b2, w->width, w->height);
		
		//Apply the derivative of the sigmoid function to the values, store it in buffer_1.
		mat_apply_function(v, b1, &sigmoidf_deriv);

		//Multiply element wise.
		mat_element_wise_mult(b1, b3, b1);
		//buffer_1 now contains the derivative with respect to the bias.

		//Apply biases.
		mat_subtract_matrix(b, b1, b);
		
		mat_transpose(b1);		
		//Calculate the derivative with respect to the weights. Store it in buffer_2.
		mat_mult_matrix(vp, b1, b2);

		//Apply weights.
		mat_subtract_matrix(w, b2, w);
		
		//Calculate next layer's error.
		mat_resize_unsafe(b3, vp->width, vp->height);
		mat_mult_matrix(b1, w, b3);
    mat_subtract_matrix(vp, b3, b3);
		mat_transpose(b1);
	}
}

float sigmoidf_deriv(float n) {
	return n * (1 - n);
}

float sigmoidf(float n) {
    return (1 / (1 + powl(EULER_NUMBER_F, -n)));
}

//Print a layer from the network.
void net_print_debug(SDLNet* net, int layer){
	if(layer >= net->layer_count) return;
	printf("Layer: %i\n", layer);
	printf("Layer count: %i\n", net->layer_count);
	printf("Layer sizes: ");
	for(int i = 0; i < net->layer_count; i++) printf("%i ", net->layer_sizes[i]);

	printf("\nBuffer_1 (derivative wrt biases):\n");
	mat_print(net->buffer_1);
	printf("Buffer_2 (derivative wrt weights):\n");
	mat_print(net->buffer_2);
	printf("Buffer_3 (error for next layer):\n");
	mat_print(net->buffer_3);

	printf("Values previous layer:\n");
	mat_print(net->values + layer - 1);
	printf("Values next layer:\n");
	mat_print(net->values + layer);
	printf("Biases:\n");
	mat_print(net->biases + layer - 1);
	printf("Weights:\n");
	mat_print(net->weights + layer - 1);
	printf("\n\n\n\n");
}
