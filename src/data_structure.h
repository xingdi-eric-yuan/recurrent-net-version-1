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
    Mat W;  // weight between current time t with previous time t-1
    Mat bW;
    Mat U;  // weight between hidden layer with previous layer
    Mat bU;
    Mat Wgrad;
    Mat bWgrad;
    Mat Ugrad;
    Mat bUgrad;
    Mat Wd2;
    Mat bWd2;
    Mat Ud2;
    Mat bUd2;
    double lr_bW;
    double lr_W;
    double lr_bU;
    double lr_U;
}Hl;

typedef struct SoftmaxRegession{
    Mat W;
    Mat b;
    Mat Wgrad;
    Mat bgrad;
    double cost;
    Mat Wd2;
    Mat bd2;
    double lr_b;
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