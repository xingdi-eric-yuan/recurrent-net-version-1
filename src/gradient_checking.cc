#include "gradient_checking.h"

using namespace cv;
using namespace std;


void
gradientChecking_SoftmaxLayer(std::vector<Hl> &hLayers, Smr &smr, std::vector<Mat> &sampleX, Mat &sampleY){
    //Gradient Checking (remember to disable this part after you're sure the 
    //cost function and dJ function are correct)
    getNetworkCost(sampleX, sampleY, hLayers, smr);
    Mat grad;
    smr.Wgrad.copyTo(grad);
    cout<<"################################################"<<endl;
    cout<<"## test softmax layer !!!!"<<endl;
    cout<<"################################################"<<endl;
    double epsilon = 1e-4;
    for(int i = 0; i < smr.W.rows; i++){
        for(int j = 0; j < smr.W.cols; j++){
            double memo = smr.W.ATD(i, j);
            smr.W.ATD(i, j) = memo + epsilon;
            getNetworkCost(sampleX, sampleY, hLayers, smr);
            double value1 = smr.cost;
            smr.W.ATD(i, j) = memo - epsilon;
            getNetworkCost(sampleX, sampleY, hLayers, smr);
            double value2 = smr.cost;
            double tp = (value1 - value2) / (2 * epsilon);
            if(tp == 0.0 && grad.ATD(i, j) == 0.0) cout<<i<<", "<<j<<", "<<tp<<", "<<grad.ATD(i, j)<<", "<<1<<endl;
            else cout<<i<<", "<<j<<", "<<tp<<", "<<grad.ATD(i, j)<<", "<<tp / grad.ATD(i, j)<<endl;
            smr.W.ATD(i, j) = memo;
        }
    }
    grad.release();
}

void
gradientChecking_HiddenLayer(std::vector<Hl> &hLayers, Smr &smr, std::vector<Mat> &sampleX, Mat &sampleY, int layer){
    //Gradient Checking (remember to disable this part after you're sure the 
    //cost function and dJ function are correct)
    getNetworkCost(sampleX, sampleY, hLayers, smr);
    Mat grad;
    double epsilon = 1e-4;
    
    hLayers[layer].Wgrad.copyTo(grad);
    cout<<"################################################"<<endl;
    cout<<"## test full-connected layer["<<layer<<"] W !!!!"<<endl;
    cout<<"################################################"<<endl;
    for(int i = 0; i < hLayers[layer].W.rows; i++){
        for(int j = 0; j < hLayers[layer].W.cols; j++){
            double memo = hLayers[layer].W.ATD(i, j);
            hLayers[layer].W.ATD(i, j) = memo + epsilon;
            getNetworkCost(sampleX, sampleY, hLayers, smr);
            double value1 = smr.cost;
            hLayers[layer].W.ATD(i, j) = memo - epsilon;
            getNetworkCost(sampleX, sampleY, hLayers, smr);
            double value2 = smr.cost;
            double tp = (value1 - value2) / (2 * epsilon);
            if(tp == 0.0 && grad.ATD(i, j) == 0.0) ;//cout<<i<<", "<<j<<", "<<tp<<", "<<grad.ATD(i, j)<<", "<<1<<endl;
            else cout<<i<<", "<<j<<", "<<tp<<", "<<grad.ATD(i, j)<<", "<<tp / grad.ATD(i, j)<<endl;
            hLayers[layer].W.ATD(i, j) = memo;
        }
    }
    hLayers[layer].Ugrad.copyTo(grad);
    cout<<"################################################"<<endl;
    cout<<"## test full-connected layer["<<layer<<"] U !!!!"<<endl;
    cout<<"################################################"<<endl;
    for(int i = 0; i < hLayers[layer].U.rows; i++){
        for(int j = 0; j < hLayers[layer].U.cols; j++){
            double memo = hLayers[layer].U.ATD(i, j);
            hLayers[layer].U.ATD(i, j) = memo + epsilon;
            getNetworkCost(sampleX, sampleY, hLayers, smr);
            double value1 = smr.cost;
            hLayers[layer].U.ATD(i, j) = memo - epsilon;
            getNetworkCost(sampleX, sampleY, hLayers, smr);
            double value2 = smr.cost;
            double tp = (value1 - value2) / (2 * epsilon);
            if(tp == 0.0 && grad.ATD(i, j) == 0.0) ;//cout<<i<<", "<<j<<", "<<tp<<", "<<grad.ATD(i, j)<<", "<<1<<endl;
            else cout<<i<<", "<<j<<", "<<tp<<", "<<grad.ATD(i, j)<<", "<<tp / grad.ATD(i, j)<<endl;
            hLayers[layer].U.ATD(i, j) = memo;
        }
    }
    grad.release();
}
