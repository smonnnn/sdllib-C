#include <stdio.h>
#include "sdllib.h"

int main(){
    printf("Starting!\n");
    SDLNet network;
    int layer_sizes[] = {3, 3, 5, 2};
    init_network(&network, layer_sizes, 4);

    Matrix input = mat_new(1, 3);
    mat_set(&input, 0, 0, 0.5f);
    mat_set(&input, 0, 1, 0.1f);
    mat_set(&input, 0, 2, 0.9f);

    float t[] = {1.0f, 1.0f};
    Matrix target = mat_new_from_data(1, 2, t);

    printf("Training!\n");
    backward(&network, &input, &target);
    
    printf("END\n");
    return 1;
}