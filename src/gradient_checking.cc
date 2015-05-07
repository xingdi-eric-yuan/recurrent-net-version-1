#include "gradient_checking.h"

using namespace cv;
using namespace std;


void
gradientChecking_SoftmaxLayer(std::vector<Hl> &hLayers, Smr &smr, std::vector<Mat> &sampleX, Mat &sampleY){
    //Gradient Checking (remember to disable this part after you're sure the 
    //cost function and dJ function are correct)
    double epsilon = 1e-4;
    getNetworkCost(sampleX, sampleY, hLayers, smr);
    Mat grad;
    smr.W_lgrad.copyTo(grad);
    cout<<"################################################"<<endl;
    cout<<"## test softmax layer --- forward !!!!"<<endl;
    cout<<"################################################"<<endl;
    for(int i = 0; i < smr.W_l.rows; i++){
        for(int j = 0; j < smr.W_l.cols; j++){
            double memo = smr.W_l.ATD(i, j);
            smr.W_l.ATD(i, j) = memo + epsilon;
            getNetworkCost(sampleX, sampleY, hLayers, smr);
            double value1 = smr.cost;
            smr.W_l.ATD(i, j) = memo - epsilon;
            getNetworkCost(sampleX, sampleY, hLayers, smr);
            double value2 = smr.cost;
            double tp = (value1 - value2) / (2 * epsilon);
            if(tp == 0.0 && grad.ATD(i, j) == 0.0) cout<<i<<", "<<j<<", "<<tp<<", "<<grad.ATD(i, j)<<", "<<1<<endl;
            else cout<<i<<", "<<j<<", "<<tp<<", "<<grad.ATD(i, j)<<", "<<tp / grad.ATD(i, j)<<endl;
            smr.W_l.ATD(i, j) = memo;
        }
    }

    smr.W_rgrad.copyTo(grad);
    cout<<"################################################"<<endl;
    cout<<"## test softmax layer --- backward !!!!"<<endl;
    cout<<"################################################"<<endl;
    for(int i = 0; i < smr.W_r.rows; i++){
        for(int j = 0; j < smr.W_r.cols; j++){
            double memo = smr.W_r.ATD(i, j);
            smr.W_r.ATD(i, j) = memo + epsilon;
            getNetworkCost(sampleX, sampleY, hLayers, smr);
            double value1 = smr.cost;
            smr.W_r.ATD(i, j) = memo - epsilon;
            getNetworkCost(sampleX, sampleY, hLayers, smr);
            double value2 = smr.cost;
            double tp = (value1 - value2) / (2 * epsilon);
            if(tp == 0.0 && grad.ATD(i, j) == 0.0) cout<<i<<", "<<j<<", "<<tp<<", "<<grad.ATD(i, j)<<", "<<1<<endl;
            else cout<<i<<", "<<j<<", "<<tp<<", "<<grad.ATD(i, j)<<", "<<tp / grad.ATD(i, j)<<endl;
            smr.W_r.ATD(i, j) = memo;
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
    
    hLayers[layer].W_lgrad.copyTo(grad);
    cout<<"################################################"<<endl;
    cout<<"## test full-connected layer["<<layer<<"] W --- forward !!!!"<<endl;
    cout<<"################################################"<<endl;
    for(int i = 0; i < hLayers[layer].W_l.rows; i++){
        for(int j = 0; j < hLayers[layer].W_l.cols; j++){
            double memo = hLayers[layer].W_l.ATD(i, j);
            hLayers[layer].W_l.ATD(i, j) = memo + epsilon;
            getNetworkCost(sampleX, sampleY, hLayers, smr);
            double value1 = smr.cost;
            hLayers[layer].W_l.ATD(i, j) = memo - epsilon;
            getNetworkCost(sampleX, sampleY, hLayers, smr);
            double value2 = smr.cost;
            double tp = (value1 - value2) / (2 * epsilon);
            if(tp == 0.0 && grad.ATD(i, j) == 0.0) ;//cout<<i<<", "<<j<<", "<<tp<<", "<<grad.ATD(i, j)<<", "<<1<<endl;
            else cout<<i<<", "<<j<<", "<<tp<<", "<<grad.ATD(i, j)<<", "<<tp / grad.ATD(i, j)<<endl;
            hLayers[layer].W_l.ATD(i, j) = memo;
        }
    }
    hLayers[layer].U_lgrad.copyTo(grad);
    cout<<"################################################"<<endl;
    cout<<"## test full-connected layer["<<layer<<"] U --- forward !!!!"<<endl;
    cout<<"################################################"<<endl;
    for(int i = 0; i < hLayers[layer].U_l.rows; i++){
        for(int j = 0; j < hLayers[layer].U_l.cols; j++){
            double memo = hLayers[layer].U_l.ATD(i, j);
            hLayers[layer].U_l.ATD(i, j) = memo + epsilon;
            getNetworkCost(sampleX, sampleY, hLayers, smr);
            double value1 = smr.cost;
            hLayers[layer].U_l.ATD(i, j) = memo - epsilon;
            getNetworkCost(sampleX, sampleY, hLayers, smr);
            double value2 = smr.cost;
            double tp = (value1 - value2) / (2 * epsilon);
            if(tp == 0.0 && grad.ATD(i, j) == 0.0) ;//cout<<i<<", "<<j<<", "<<tp<<", "<<grad.ATD(i, j)<<", "<<1<<endl;
            else cout<<i<<", "<<j<<", "<<tp<<", "<<grad.ATD(i, j)<<", "<<tp / grad.ATD(i, j)<<endl;
            hLayers[layer].U_l.ATD(i, j) = memo;
        }
    }

    hLayers[layer].W_rgrad.copyTo(grad);
    cout<<"################################################"<<endl;
    cout<<"## test full-connected layer["<<layer<<"] W --- backward !!!!"<<endl;
    cout<<"################################################"<<endl;
    for(int i = 0; i < hLayers[layer].W_r.rows; i++){
        for(int j = 0; j < hLayers[layer].W_r.cols; j++){
            double memo = hLayers[layer].W_r.ATD(i, j);
            hLayers[layer].W_r.ATD(i, j) = memo + epsilon;
            getNetworkCost(sampleX, sampleY, hLayers, smr);
            double value1 = smr.cost;
            hLayers[layer].W_r.ATD(i, j) = memo - epsilon;
            getNetworkCost(sampleX, sampleY, hLayers, smr);
            double value2 = smr.cost;
            double tp = (value1 - value2) / (2 * epsilon);
            if(tp == 0.0 && grad.ATD(i, j) == 0.0) ;//cout<<i<<", "<<j<<", "<<tp<<", "<<grad.ATD(i, j)<<", "<<1<<endl;
            else cout<<i<<", "<<j<<", "<<tp<<", "<<grad.ATD(i, j)<<", "<<tp / grad.ATD(i, j)<<endl;
            hLayers[layer].W_r.ATD(i, j) = memo;
        }
    }
    hLayers[layer].U_rgrad.copyTo(grad);
    cout<<"################################################"<<endl;
    cout<<"## test full-connected layer["<<layer<<"] U --- backward !!!!"<<endl;
    cout<<"################################################"<<endl;
    for(int i = 0; i < hLayers[layer].U_r.rows; i++){
        for(int j = 0; j < hLayers[layer].U_r.cols; j++){
            double memo = hLayers[layer].U_r.ATD(i, j);
            hLayers[layer].U_r.ATD(i, j) = memo + epsilon;
            getNetworkCost(sampleX, sampleY, hLayers, smr);
            double value1 = smr.cost;
            hLayers[layer].U_r.ATD(i, j) = memo - epsilon;
            getNetworkCost(sampleX, sampleY, hLayers, smr);
            double value2 = smr.cost;
            double tp = (value1 - value2) / (2 * epsilon);
            if(tp == 0.0 && grad.ATD(i, j) == 0.0) ;//cout<<i<<", "<<j<<", "<<tp<<", "<<grad.ATD(i, j)<<", "<<1<<endl;
            else cout<<i<<", "<<j<<", "<<tp<<", "<<grad.ATD(i, j)<<", "<<tp / grad.ATD(i, j)<<endl;
            hLayers[layer].U_r.ATD(i, j) = memo;
        }
    }
    grad.release();
}

