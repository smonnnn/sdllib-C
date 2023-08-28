#include <stdio.h>
#include <stdlib.h>
#include "sdllib.h"
#include <time.h>

int main(){
    srand(time(NULL));
    printf("Starting!\n");
    SDLNet network;
    int layer_sizes[] = {3, 10, 50, 100, 50, 10, 2};
    init_network(&network, layer_sizes, 7);

    Matrix input = mat_new(1, 3);
    mat_set(&input, 0, 0, 0.5f);
    mat_set(&input, 0, 1, 0.1f);
    mat_set(&input, 0, 2, 0.9f);

    float t[] = {0.5f, 0.5f};
    Matrix target = mat_new_from_data(1, 2, t);

    printf("Training!\n");
    int i = 0;
    
    while(network.total_error > 0.01f){
        backward(&network, &input, &target);
        printf("%4i error: %7.2f\n", i, network.total_error);
        i++;
    }
    
    printf("END\n");

    delete_network(&network);
    return 1;
}