#include <stdio.h>
#include <assert.h>
#include "net.h"

int main(){
    printf("Starting!\n");
    int ls[] = {5, 300, 300, 100, 20, 3};
    float input[] = {rand_01(), rand_01(), rand_01(), rand_01(), rand_01()};
    float target[] = {rand_01(), rand_01(), rand_01()};
    
    Network net = create_network(ls, 6);

    printf("Running the network!\n");
    feed_forward(net, input);

    printf("Training!\n");

    float e;
    e = train(net, input, target);
    printf("Error: %f\n", e);
    while(e > 0.01){
        e = train(net, input, target);
        printf("Error: %f\n", e);
    }
    
    printf("END\n");
    return 1;
}