#include "weight_init.h"

using namespace cv;
using namespace std;
void
weightRandomInit(Hl &ntw, int inputsize, int hiddensize){
    double epsilon = 0.12;
    // weight between hidden layer with previous layer
    ntw.U_l = Mat::ones(hiddensize, inputsize, CV_64FC1);
    randu(ntw.U_l, Scalar(-1.0), Scalar(1.0));
    ntw.U_l = ntw.U_l * epsilon;
    ntw.U_lgrad = Mat::zeros(hiddensize, inputsize, CV_64FC1);
    ntw.U_ld2 = Mat::zeros(ntw.U_l.size(), CV_64FC1);
    ntw.lr_U = lrate_w;
    
    ntw.W_l = Mat::ones(hiddensize, hiddensize, CV_64FC1);
    randu(ntw.W_l, Scalar(-1.0), Scalar(1.0));
    ntw.W_l = ntw.W_l * epsilon;
    ntw.W_lgrad = Mat::zeros(hiddensize, hiddensize, CV_64FC1);
    ntw.W_ld2 = Mat::zeros(ntw.W_l.size(), CV_64FC1);
    ntw.lr_W = lrate_w;

    ntw.U_r = Mat::ones(hiddensize, inputsize, CV_64FC1);
    randu(ntw.U_r, Scalar(-1.0), Scalar(1.0));
    ntw.U_r = ntw.U_r * epsilon;
    ntw.U_rgrad = Mat::zeros(hiddensize, inputsize, CV_64FC1);
    ntw.U_rd2 = Mat::zeros(ntw.U_r.size(), CV_64FC1);
    ntw.lr_U = lrate_w;
    
    ntw.W_r = Mat::ones(hiddensize, hiddensize, CV_64FC1);
    randu(ntw.W_r, Scalar(-1.0), Scalar(1.0));
    ntw.W_r = ntw.W_r * epsilon;
    ntw.W_rgrad = Mat::zeros(hiddensize, hiddensize, CV_64FC1);
    ntw.W_rd2 = Mat::zeros(ntw.W_r.size(), CV_64FC1);
    ntw.lr_W = lrate_w;
}

void 
weightRandomInit(Smr &smr, int nclasses, int nfeatures){
    double epsilon = 0.12;
    smr.W_l = Mat::ones(nclasses, nfeatures, CV_64FC1);
    randu(smr.W_l, Scalar(-1.0), Scalar(1.0));
    smr.W_l = smr.W_l * epsilon;
    smr.W_lgrad = Mat::zeros(nclasses, nfeatures, CV_64FC1);
    smr.W_ld2 = Mat::zeros(smr.W_l.size(), CV_64FC1);

    smr.W_r = Mat::ones(nclasses, nfeatures, CV_64FC1);
    randu(smr.W_r, Scalar(-1.0), Scalar(1.0));
    smr.W_r = smr.W_r * epsilon;
    smr.W_rgrad = Mat::zeros(nclasses, nfeatures, CV_64FC1);
    smr.W_rd2 = Mat::zeros(smr.W_r.size(), CV_64FC1);
    
    smr.cost = 0.0;
    smr.lr_W = lrate_w;
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


