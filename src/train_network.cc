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
        Mat sampleY = Mat::zeros(nGram, batch_size, CV_64FC1);
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
        Mat v_smr_W_l = Mat::zeros(smr.W_l.size(), CV_64FC1);
        Mat smrW_ld2 = Mat::zeros(smr.W_l.size(), CV_64FC1);
        Mat v_smr_W_r = Mat::zeros(smr.W_l.size(), CV_64FC1);
        Mat smrW_rd2 = Mat::zeros(smr.W_l.size(), CV_64FC1);
        std::vector<Mat> v_hl_W_l;
        std::vector<Mat> hlW_ld2;
        std::vector<Mat> v_hl_U_l;
        std::vector<Mat> hlU_ld2;
        std::vector<Mat> v_hl_W_r;
        std::vector<Mat> hlW_rd2;
        std::vector<Mat> v_hl_U_r;
        std::vector<Mat> hlU_rd2;
        for(int i = 0; i < HiddenLayers.size(); ++i){
            Mat tmpW = Mat::zeros(HiddenLayers[i].W_l.size(), CV_64FC1);
            Mat tmpU = Mat::zeros(HiddenLayers[i].U_l.size(), CV_64FC1);
            v_hl_W_l.push_back(tmpW);
            v_hl_U_l.push_back(tmpU);
            hlW_ld2.push_back(tmpW);
            hlU_ld2.push_back(tmpU);
            v_hl_W_r.push_back(tmpW);
            v_hl_U_r.push_back(tmpU);
            hlW_rd2.push_back(tmpW);
            hlU_rd2.push_back(tmpU);
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
                Mat sampleY = Mat::zeros(nGram, batch_size, CV_64FC1);
                getSample(x, sampleX, y, sampleY, re_wordmap);
                getNetworkCost(sampleX, sampleY, HiddenLayers, smr);
                // softmax update
                smrW_ld2 = Momentum_d2 * smrW_ld2 + (1.0 - Momentum_d2) * smr.W_ld2;
                lr_W = smr.lr_W / (smrW_ld2 + mu);
                v_smr_W_l = v_smr_W_l * Momentum_w + (1.0 - Momentum_w) * smr.W_lgrad.mul(lr_W);
                smr.W_l -= v_smr_W_l;

                smrW_rd2 = Momentum_d2 * smrW_rd2 + (1.0 - Momentum_d2) * smr.W_rd2;
                lr_W = smr.lr_W / (smrW_rd2 + mu);
                v_smr_W_r = v_smr_W_r * Momentum_w + (1.0 - Momentum_w) * smr.W_rgrad.mul(lr_W);
                smr.W_r -= v_smr_W_r;

                // hidden layer update
                for(int i = 0; i < HiddenLayers.size(); i++){
                    hlW_ld2[i] = Momentum_d2 * hlW_ld2[i] + (1.0 - Momentum_d2) * HiddenLayers[i].W_ld2;
                    hlU_ld2[i] = Momentum_d2 * hlU_ld2[i] + (1.0 - Momentum_d2) * HiddenLayers[i].U_ld2;
                    lr_W = HiddenLayers[i].lr_W / (hlW_ld2[i] + mu);
                    lr_U = HiddenLayers[i].lr_U / (hlU_ld2[i] + mu);
                    v_hl_W_l[i] = v_hl_W_l[i] * Momentum_w + (1.0 - Momentum_w) * HiddenLayers[i].W_lgrad.mul(lr_W);
                    v_hl_U_l[i] = v_hl_U_l[i] * Momentum_u + (1.0 - Momentum_u) * HiddenLayers[i].U_lgrad.mul(lr_U);
                    HiddenLayers[i].W_l -= v_hl_W_l[i];
                    HiddenLayers[i].U_l -= v_hl_U_l[i];

                    hlW_rd2[i] = Momentum_d2 * hlW_rd2[i] + (1.0 - Momentum_d2) * HiddenLayers[i].W_rd2;
                    hlU_rd2[i] = Momentum_d2 * hlU_rd2[i] + (1.0 - Momentum_d2) * HiddenLayers[i].U_rd2;
                    lr_W = HiddenLayers[i].lr_W / (hlW_rd2[i] + mu);
                    lr_U = HiddenLayers[i].lr_U / (hlU_rd2[i] + mu);
                    v_hl_W_r[i] = v_hl_W_r[i] * Momentum_w + (1.0 - Momentum_w) * HiddenLayers[i].W_rgrad.mul(lr_W);
                    v_hl_U_r[i] = v_hl_U_r[i] * Momentum_u + (1.0 - Momentum_u) * HiddenLayers[i].U_rgrad.mul(lr_U);
                    HiddenLayers[i].W_r -= v_hl_W_r[i];
                    HiddenLayers[i].U_r -= v_hl_U_r[i];
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
        v_smr_W_l.release();
        v_hl_W_l.clear();
        std::vector<Mat>().swap(v_hl_W_l);
        v_hl_U_l.clear();
        std::vector<Mat>().swap(v_hl_U_l);
        hlW_ld2.clear();
        std::vector<Mat>().swap(hlW_ld2);
        hlU_ld2.clear();
        std::vector<Mat>().swap(hlU_ld2);
        v_smr_W_r.release();
        v_hl_W_r.clear();
        std::vector<Mat>().swap(v_hl_W_r);
        v_hl_U_r.clear();
        std::vector<Mat>().swap(v_hl_U_r);
        hlW_rd2.clear();
        std::vector<Mat>().swap(hlW_rd2);
        hlU_rd2.clear();
        std::vector<Mat>().swap(hlU_rd2);
        
    }
}




