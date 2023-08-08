#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "net.h"

float train(Network net, float* input, float* target){
    int last_layer_size = net.layers[net.layer_count - 1].size;
    feed_forward(net, input);
    float total_error = calc_total_error(target, net.layers[net.layer_count - 1].values, last_layer_size);

    float* error = malloc(last_layer_size * sizeof(float));
    for(int i = 0; i < last_layer_size; i++){
        error[i] = net.layers[net.layer_count - 1].values[i] - target[i];
    }


    //Should I rewrite this to apply the changes directly to the biases? Or keep it like this to make changing to opencl easier?
    float* d_biases;
    for(int layer = net.layer_count - 1; layer > 0; layer--){
        d_biases = d_bias(net.layers[layer].values, error, net.layers[layer].size);
        d_weight(d_biases, net.layers[layer - 1].values, net.layers[layer - 1].weights, net.layers[layer - 1].size, net.layers[layer].size);
        free(error);
        error = calc_error(d_biases, net.layers[layer - 1].weights, net.layers[layer - 1].size, net.layers[layer].size);
        for(int b = 0; b < net.layers[layer].size; b++){
            net.layers[layer].biases[b] -= d_biases[b];
        }
        free(d_biases);
    }
    return total_error;
}

//Feeds forward through the network
void feed_forward(Network net, float* input){
    net.layers[0].values = input;
    for(int layer = 0; layer < net.layer_count - 1; layer++){
        calc_layer(net.layers[layer], net.layers[layer + 1].values, net.layers[layer + 1].size);
    }
}

/*
    This function is used to calculate the next layer's nodes values for a given layer.

    Layer layer - Current Layer
    float* output_values - Pointer to empty array of the next layer's values.
    int next_layer_size - Node count of the next layer ie next layer's size.
*/
void calc_layer(Layer layer, float* output_values, int next_layer_size){
    for(int y = 0; y < next_layer_size; y++){
        output_values[y] = layer.biases[y];
        for(int x = 0; x < layer.size; x++){
            output_values[y] += layer.values[x] * layer.weights[x][y];
        }
        output_values[y] = sigmoidf(output_values[y]);
    }
}

/*
    This function calculates and applies the derivative of the weight with respect to the error.

    float* d_biases - Calculated using float* d_bias(), these are the derivatives of the errors with respect to their biases.
    float* values - Output values from the layer before the current one. 
    float** d_weights - Pointer to the array for containing the weights going from the previous layer to the current layer, format weights[layer - 1][layer] of layers[layer - 1].
    int y_size - The current layer's node count.
    int x_size - The previous layer's node count.
*/
void d_weight(float* d_biases, float* values, float** d_weights, int x_size, int y_size){
    for(int x = 0; x < x_size; x++){
        for(int y = 0; y < y_size; y++){
            d_weights[x][y] -= values[x] * d_biases[y] * learn_rate;
        }
    }
}

/*
    This function calculates and stores but DOES NOT APPLY the derivative of the bias with respect to the errors.

    float* values - Values of the current layer's nodes.
    float* error - Error of the current layer's nodes.
    int size - size of all input arrays, ie the current layer's node count.
*/
float* d_bias(float* values, float* error, int size){
    float* d_biases = malloc(size * sizeof(float));
    for(int x = 0; x < size; x++){
        d_biases[x] = error[x] * values[x] * (1 - values[x]) * learn_rate;
        //printf("V%.2fE%.2fB%.2f\n", values[x], error[x], d_biases[x]);
    }
    return d_biases;
}
/*
    This function calculates the error for the next layer for backpropagation.

    float* d_biases - Calculated using float* d_bias(), these are the derivatives of the errors with respect to their biases.
    float** weights - Matrix of weights used for calculating the current layer (likely layer_index - 1 due to last layer not having weights), matrix is arranged like this: weights[previous layer node][current layer node], and resembles the connection between the two layer's nodes.
    int x_size - Size of the previous layer, used to allocate the output array.
    int y_size - Size of the current layer.
*/
float* calc_error(float* d_biases, float** weights, int x_size, int y_size){
    float* error = malloc(x_size * sizeof(float));
    for(int x = 0; x < x_size; x++){
        error[x] = 0;
        for(int y = 0; y < y_size; y++){
            error[x] += d_biases[y] * weights[x][y];
        }
    }
    return error;
}

//This function calculates the error for the entire network given an output and target output
float calc_total_error(float* target, float* output, int size){
    float total_error = 0;
    for(int i = 0; i < size; i++){
        total_error += 0.5 * pow(target[i] - output[i], 2);
    }
    return total_error;
}

float sigmoidf(float n) {
    return (1 / (1 + powl(EULER_NUMBER_F, -n)));
}

float rand_01(){
    return rand() / ((double) RAND_MAX);
}

Network create_network(int* layer_sizes, int layer_count){
    srand(time(0));
    Network net;
    net.layer_count = layer_count;
    net.layers = malloc(layer_count * sizeof(Layer));

    for(int layer = 0; layer < layer_count; layer++){
        net.layers[layer].size = layer_sizes[layer];
        net.layers[layer].biases = malloc(layer_sizes[layer] * sizeof(float));
        net.layers[layer].values = malloc(layer_sizes[layer] * sizeof(float));
        net.layers[layer].weights = malloc(layer_sizes[layer] * sizeof(float*));

        for(int node = 0; node < layer_sizes[layer]; node++){
            net.layers[layer].weights[node] = malloc(layer_sizes[layer + 1] * sizeof(float));
            net.layers[layer].biases[node] = rand_01();
            
            for(int next_node = 0; (next_node < layer_sizes[layer + 1]) && layer != (layer_count - 1); next_node++){
                net.layers[layer].weights[node][next_node] = rand_01();
            }
        }
    }
    return net;
}
/*

*/