#define EULER_NUMBER_F 2.71828182846
#define learn_rate 1.0f

typedef struct Layer{
    int size;
    float* values;
    float* biases;
    float** weights;
} Layer;

typedef struct Network{
    int layer_count;
    Layer* layers;
} Network;



void feed_forward(Network net, float* input);
void calc_layer(Layer layer, float* output_values, int next_layer_size);

float train(Network net, float* input, float* target);
void d_weight(float* d_biases, float* values, float** d_weights, int x_size, int y_size);
float* d_bias(float* values, float* error, int size);
float* calc_error(float* d_biases, float** weights, int x_size, int y_size);

float calc_total_error(float* target, float* output, int size);

Network create_network(int* layer_sizes, int layer_count);
float sigmoidf(float n);
float rand_01();
