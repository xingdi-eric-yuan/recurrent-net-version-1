#include "weight_init.h"

using namespace cv;
using namespace std;
void
weightRandomInit(Hl &ntw, int inputsize, int hiddensize){
    double epsilon = 0.12;
    // weight between hidden layer with previous layer
    ntw.U = Mat::ones(hiddensize, inputsize, CV_64FC1);
    randu(ntw.U, Scalar(-1.0), Scalar(1.0));
    ntw.U = ntw.U * epsilon;
    ntw.bU = Mat::zeros(hiddensize, 1, CV_64FC1);
    ntw.Ugrad = Mat::zeros(hiddensize, inputsize, CV_64FC1);
    ntw.bUgrad = Mat::zeros(hiddensize, 1, CV_64FC1);
    ntw.Ud2 = Mat::zeros(ntw.U.size(), CV_64FC1);
    ntw.bUd2 = Mat::zeros(ntw.bU.size(), CV_64FC1);
    ntw.lr_U = lrate_w;
    ntw.lr_bU = lrate_b;

    ntw.W = Mat::ones(hiddensize, hiddensize, CV_64FC1);
    randu(ntw.W, Scalar(-1.0), Scalar(1.0));
    ntw.W = ntw.W * epsilon;
    ntw.bW = Mat::zeros(hiddensize, 1, CV_64FC1);
    ntw.Wgrad = Mat::zeros(hiddensize, hiddensize, CV_64FC1);
    ntw.bWgrad = Mat::zeros(hiddensize, 1, CV_64FC1);
    ntw.Wd2 = Mat::zeros(ntw.W.size(), CV_64FC1);
    ntw.bWd2 = Mat::zeros(ntw.bW.size(), CV_64FC1);
    ntw.lr_W = lrate_w;
    ntw.lr_bW = lrate_b;
}

void 
weightRandomInit(Smr &smr, int nclasses, int nfeatures){
    double epsilon = 0.12;
    smr.W = Mat::ones(nclasses, nfeatures, CV_64FC1);
    randu(smr.W, Scalar(-1.0), Scalar(1.0));
    smr.W = smr.W * epsilon;
    smr.b = Mat::zeros(nclasses, 1, CV_64FC1);
    smr.cost = 0.0;
    smr.Wgrad = Mat::zeros(nclasses, nfeatures, CV_64FC1);
    smr.bgrad = Mat::zeros(nclasses, 1, CV_64FC1);
    smr.Wd2 = Mat::zeros(smr.W.size(), CV_64FC1);
    smr.bd2 = Mat::zeros(smr.b.size(), CV_64FC1);
    smr.lr_W = lrate_w;
    smr.lr_b = lrate_b;
}

void
rnnInitPrarms(std::vector<Hl> &HiddenLayers, Smr &smr){
    
    // Init Hidden layers
    if(hiddenConfig.size() > 0){
        Hl tpntw; 
        weightRandomInit(tpntw, word_vec_len, hiddenConfig[0].NumHiddenNeurons);
        HiddenLayers.push_back(tpntw);
        for(int i = 1; i < hiddenConfig.size(); i++){
            Hl tpntw2;
            weightRandomInit(tpntw2, hiddenConfig[i - 1].NumHiddenNeurons, hiddenConfig[i].NumHiddenNeurons);
            HiddenLayers.push_back(tpntw2);
        }
    }
    // Init Softmax layer
    if(hiddenConfig.size() == 0){
        weightRandomInit(smr, softmaxConfig.NumClasses, word_vec_len);
    }else{
        weightRandomInit(smr, softmaxConfig.NumClasses, hiddenConfig[hiddenConfig.size() - 1].NumHiddenNeurons);
    }
}


