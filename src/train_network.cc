#include "train_network.h"

using namespace cv;
using namespace std;
void
trainNetwork(const std::vector<std::vector<int> > &x, std::vector<std::vector<int> > &y, std::vector<Hl> &HiddenLayers, Smr &smr, 
             const std::vector<std::vector<int> > &tx, std::vector<std::vector<int> > &ty, std::vector<string> &re_wordmap
             ){
    if (is_gradient_checking){
        batch_size = 2;
        std::vector<Mat> sampleX;
        Mat sampleY = Mat::zeros(1, batch_size, CV_64FC1);
        getSample(x, sampleX, y, sampleY, re_wordmap);
        for(int i = 0; i < hiddenConfig.size(); i++){
            gradientChecking_HiddenLayer(HiddenLayers, smr, sampleX, sampleY, i);   
        }
        gradientChecking_SoftmaxLayer(HiddenLayers, smr, sampleX, sampleY);
    }else{
        cout<<"****************************************************************************"<<endl
            <<"**                       TRAINING NETWORK......                             "<<endl
            <<"****************************************************************************"<<endl<<endl;
        // velocity vectors.
        Mat v_smr_W = Mat::zeros(smr.W.size(), CV_64FC1);
        Mat smrWd2 = Mat::zeros(smr.W.size(), CV_64FC1);
        std::vector<Mat> v_hl_W;
        std::vector<Mat> hlWd2;
        std::vector<Mat> v_hl_U;
        std::vector<Mat> hlUd2;
        for(int i = 0; i < HiddenLayers.size(); ++i){
            Mat tmpW = Mat::zeros(HiddenLayers[i].W.size(), CV_64FC1);
            Mat tmpU = Mat::zeros(HiddenLayers[i].U.size(), CV_64FC1);
            v_hl_W.push_back(tmpW);
            v_hl_U.push_back(tmpU);
            hlWd2.push_back(tmpW);
            hlUd2.push_back(tmpU);
        }

        double Momentum_w = 0.5;
        double Momentum_u = 0.5;
        double Momentum_d2 = 0.5;
        Mat lr_W;
        Mat lr_U;
        double mu = 1e-2;
        int k = 0;

        for(int epo = 1; epo <= training_epochs; epo++){
            for(; k <= iter_per_epo * epo; k++){
                if(k > 30) {Momentum_w = 0.95; Momentum_u = 0.95; Momentum_d2 = 0.90;}
                cout<<"epoch: "<<epo<<", iter: "<<k;//<<endl;     
                std::vector<Mat> sampleX;
                Mat sampleY = Mat::zeros(1, batch_size, CV_64FC1);
                getSample(x, sampleX, y, sampleY, re_wordmap);
                getNetworkCost(sampleX, sampleY, HiddenLayers, smr);
                // softmax update
                smrWd2 = Momentum_d2 * smrWd2 + (1.0 - Momentum_d2) * smr.Wd2;
                lr_W = smr.lr_W / (smrWd2 + mu);
                v_smr_W = v_smr_W * Momentum_w + (1.0 - Momentum_w) * smr.Wgrad.mul(lr_W);
                smr.W -= v_smr_W;
                // hidden layer update
                for(int i = 0; i < HiddenLayers.size(); i++){
                    hlWd2[i] = Momentum_d2 * hlWd2[i] + (1.0 - Momentum_d2) * HiddenLayers[i].Wd2;
                    hlUd2[i] = Momentum_d2 * hlUd2[i] + (1.0 - Momentum_d2) * HiddenLayers[i].Ud2;
                    lr_W = HiddenLayers[i].lr_W / (hlWd2[i] + mu);
                    lr_U = HiddenLayers[i].lr_U / (hlUd2[i] + mu);
                    v_hl_W[i] = v_hl_W[i] * Momentum_w + (1.0 - Momentum_w) * HiddenLayers[i].Wgrad.mul(lr_W);
                    v_hl_U[i] = v_hl_U[i] * Momentum_u + (1.0 - Momentum_u) * HiddenLayers[i].Ugrad.mul(lr_U);
                    HiddenLayers[i].W -= v_hl_W[i];
                    HiddenLayers[i].U -= v_hl_U[i];
                }
                sampleX.clear();
                std::vector<Mat>().swap(sampleX);
            }
            if(!is_gradient_checking){
                cout<<"Test training data: "<<endl;;
                testNetwork(x, y, HiddenLayers, smr, re_wordmap);
                cout<<"Test testing data: "<<endl;;
                testNetwork(tx, ty, HiddenLayers, smr, re_wordmap);
            }
        }
        v_smr_W.release();
        v_hl_W.clear();
        std::vector<Mat>().swap(v_hl_W);
        v_hl_U.clear();
        std::vector<Mat>().swap(v_hl_U);
        hlWd2.clear();
        std::vector<Mat>().swap(hlWd2);
        hlUd2.clear();
        std::vector<Mat>().swap(hlUd2);
    }
}




