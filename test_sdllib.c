#include <stdio.h>
#include "sdllib.h"

int main(){
    printf("Starting!\n");
    SDLNet network;
    int layer_sizes[] = {3, 3, 2, 1};
    init_network(&network, layer_sizes, 4);

    Matrix input = mat_new(3, 1);
    mat_set(&input, 0, 0, 0.5f);
    mat_set(&input, 1, 0, 0.1f);
    mat_set(&input, 2, 0, 0.9f);

    float target[] = {1.0f};

    printf("Training!\n");
    backward(&network, &input, target);
    
    printf("END\n");
    return 1;
}