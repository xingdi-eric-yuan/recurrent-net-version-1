#pragma once
#include "general_settings.h"

using namespace std;
using namespace cv;

///////////////////////////////////
// mitie Structures
///////////////////////////////////
struct singleWord {
    std::string word;
    int label;
    singleWord(string a, int b) : word(a), label(b){}
};


///////////////////////////////////
// Network Layer Structures
///////////////////////////////////

typedef struct HiddenLayer{
    Mat W_l;  // weight between current time t with previous time t-1
    Mat U_l;  // weight between hidden layer with previous layer
    Mat W_lgrad;
    Mat U_lgrad;
    Mat W_ld2;
    Mat U_ld2;
    Mat W_r;  // weight between current time t with previous time t-1
    Mat U_r;  // weight between hidden layer with previous layer
    Mat W_rgrad;
    Mat U_rgrad;
    Mat W_rd2;
    Mat U_rd2;
    double lr_W;
    double lr_U;
}Hl;

typedef struct SoftmaxRegession{
    Mat W_l;
    Mat W_lgrad;
    Mat W_ld2;
    Mat W_r;
    Mat W_rgrad;
    Mat W_rd2;
    double cost;
    double lr_W;
}Smr;

///////////////////////////////////
// Config Structures
///////////////////////////////////
///
struct HiddenLayerConfig {
    int NumHiddenNeurons;
    double WeightDecay;
    double DropoutRate;
    HiddenLayerConfig(int a, double b, double c) : NumHiddenNeurons(a), WeightDecay(b), DropoutRate(c) {}
};

struct SoftmaxLayerConfig {
    int NumClasses;
    double WeightDecay;
    //SoftmaxLayerConfig(int a, double b) : NumClasses(a), WeightDecay(b) {}
};