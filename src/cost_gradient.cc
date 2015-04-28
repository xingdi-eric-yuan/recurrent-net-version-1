#include "cost_gradient.h"

using namespace cv;
using namespace std;
void 
getNetworkCost(std::vector<Mat> &x, Mat &y, std::vector<Hl> &hLayers, Smr &smr){

    int T = x.size();
    int nSamples = x[0].cols;
    // hidden layer forward
    std::vector<std::vector<Mat> > nonlin;
    std::vector<std::vector<Mat> > acti;
    std::vector<std::vector<Mat> > bernoulli;

    std::vector<Mat> tmp_vec;
    acti.push_back(tmp_vec);
    for(int i = 0; i < T; ++i){
        acti[0].push_back(x[i]);
    }
    for(int i = 1; i <= hiddenConfig.size(); ++i){
        // for each hidden layer
        acti.push_back(tmp_vec);
        nonlin.push_back(tmp_vec);
        bernoulli.push_back(tmp_vec);
        for(int j = 0; j < T; ++j){
            // for each time slot
            Mat tmpacti = hLayers[i - 1].U * acti[i - 1][j];
            if(j > 0) tmpacti += hLayers[i - 1].W * acti[i][j - 1];
            nonlin[i - 1].push_back(tmpacti);
            tmpacti = ReLU(tmpacti);
            if(hiddenConfig[i - 1].DropoutRate < 1.0){
                Mat bnl = getBernoulliMatrix(tmpacti.rows, tmpacti.cols, hiddenConfig[i - 1].DropoutRate);
                acti[i].push_back(tmpacti.mul(bnl));
                bernoulli[i - 1].push_back(bnl);
            }else acti[i].push_back(tmpacti);
        }
    }
    // softmax layer forward
    Mat M = smr.W * acti[acti.size() - 1][T - 1];
    M -= repeat(reduce(M, 0, CV_REDUCE_MAX), M.rows, 1);
    M = exp(M);
    Mat p = divide(M, repeat(reduce(M, 0, CV_REDUCE_SUM), M.rows, 1));
    

    Mat groundTruth = Mat::zeros(softmaxConfig.NumClasses, nSamples, CV_64FC1);
    for(int i = 0; i < nSamples; i++){
        groundTruth.ATD(y.ATD(0, i), i) = 1.0;
    }
    double J1 = - sum1(groundTruth.mul(log(p))) / nSamples;
    double J2 = sum1(pow(smr.W, 2.0)) * softmaxConfig.WeightDecay / 2;
    double J3 = 0.0; 
    double J4 = 0.0;
    for(int hl = 0; hl < hLayers.size(); hl++){
        J3 += sum1(pow(hLayers[hl].W, 2.0)) * hiddenConfig[hl].WeightDecay / 2;
    }
    for(int hl = 0; hl < hLayers.size(); hl++){
        J4 += sum1(pow(hLayers[hl].U, 2.0)) * hiddenConfig[hl].WeightDecay / 2;
    }
    smr.cost = J1 + J2 + J3 + J4;
    if(!is_gradient_checking) 
        cout<<", J1 = "<<J1<<", J2 = "<<J2<<", J3 = "<<J3<<", J4 = "<<J4<<", Cost = "<<smr.cost<<endl;

    Mat tmp, tmp2;
    tmp = - (groundTruth - p) * acti[acti.size() - 1][T - 1].t()  / nSamples;
    smr.Wgrad =  tmp + softmaxConfig.WeightDecay * smr.W;
    tmp = pow((groundTruth - p), 2.0) * pow(acti[acti.size() - 1][T - 1].t(), 2.0) / nSamples;
    smr.Wd2 = tmp + softmaxConfig.WeightDecay;

    // hidden layer backward
    std::vector<std::vector<Mat> > delta(acti.size());
    std::vector<std::vector<Mat> > deltad2(acti.size());
    for(int i = 0; i < delta.size(); i++){
        delta[i].clear();
        deltad2[i].clear();
        Mat tmpmat;
        for(int j = 0; j < T; j++){
            delta[i].push_back(tmpmat);
            deltad2[i].push_back(tmpmat);
        }
    }
    for(int i = T - 1; i >= 0; i--){
        if(i == T - 1){
            tmp = -smr.W.t() * (groundTruth - p);
            tmp2 = pow(smr.W.t(), 2.0) * pow((groundTruth - p), 2.0);
        }else{
            tmp = hLayers[hLayers.size() - 1].W.t() * delta[delta.size() - 1][i + 1];
            tmp2 = pow(hLayers[hLayers.size() - 1].W.t(), 2.0) * deltad2[deltad2.size() - 1][i + 1];
        }
        tmp.copyTo(delta[delta.size() - 1][i]);
        tmp2.copyTo(deltad2[deltad2.size() - 1][i]);
        delta[delta.size() - 1][i] = delta[delta.size() - 1][i].mul(dReLU(nonlin[nonlin.size() - 1][i]));
        deltad2[deltad2.size() - 1][i] = deltad2[deltad2.size() - 1][i].mul(pow(dReLU(nonlin[nonlin.size() - 1][i]), 2.0));
        if(hiddenConfig[hiddenConfig.size() - 1].DropoutRate < 1.0){
            delta[delta.size() - 1][i] = delta[delta.size() -1][i].mul(bernoulli[bernoulli.size() - 1][i]);
            deltad2[deltad2.size() - 1][i] = deltad2[deltad2.size() -1][i].mul(pow(bernoulli[bernoulli.size() - 1][i], 2.0));
        } 
    }

    for(int i = delta.size() - 2; i > 0; --i){
        for(int j = T - 1; j >= 0; --j){
            tmp = hLayers[i].U.t() * delta[i + 1][j];
            tmp2 = pow(hLayers[i].U.t(), 2.0) * deltad2[i + 1][j];
            if(j < T - 1){
                tmp += hLayers[i - 1].W.t() * delta[i][j + 1];
                tmp2 += pow(hLayers[i - 1].W.t(), 2.0) * deltad2[i][j + 1];
            }
            tmp.copyTo(delta[i][j]);
            tmp2.copyTo(deltad2[i][j]);
            delta[i][j] = delta[i][j].mul(dReLU(nonlin[i - 1][j]));
            deltad2[i][j] = deltad2[i][j].mul(pow(dReLU(nonlin[i - 1][j]), 2.0));
            if(hiddenConfig[i - 1].DropoutRate < 1.0){
                delta[i][j] = delta[i][j].mul(bernoulli[i - 1][j]);
                deltad2[i][j] = deltad2[i][j].mul(pow(bernoulli[i - 1][j], 2.0));
            }
        }
    }
    for(int i = hiddenConfig.size() - 1; i >= 0; i--){
        tmp = delta[i + 1][0] * acti[i][0].t();
        tmp2 = deltad2[i + 1][0] * pow(acti[i][0].t(), 2.0);
        for(int j = 1; j < T; ++j){
            tmp += delta[i + 1][j] * acti[i][j].t();
            tmp2 += deltad2[i + 1][j] * pow(acti[i][j].t(), 2.0);
        }
        hLayers[i].Ugrad = tmp / nSamples + hiddenConfig[i].WeightDecay * hLayers[i].U;
        hLayers[i].Ud2 = tmp2 / nSamples + hiddenConfig[i].WeightDecay;

        tmp = delta[i + 1][T - 1] * acti[i + 1][T - 2].t();
        tmp2 = deltad2[i + 1][T - 1] * pow(acti[i + 1][T - 2].t(), 2.0);
        for(int j = T - 2; j > 0; j--){
            tmp += delta[i + 1][j] * acti[i + 1][j - 1].t();
            tmp2 += deltad2[i + 1][j] * pow(acti[i + 1][j - 1].t(), 2.0);
        }
        hLayers[i].Wgrad = tmp / nSamples + hiddenConfig[i].WeightDecay * hLayers[i].W;
        hLayers[i].Wd2 = tmp2 / nSamples + hiddenConfig[i].WeightDecay;
    }
    // destructor
    acti.clear();
    std::vector<std::vector<Mat> >().swap(acti);
    nonlin.clear();
    std::vector<std::vector<Mat> >().swap(nonlin);
    delta.clear();
    std::vector<std::vector<Mat> >().swap(delta);
    deltad2.clear();
    std::vector<std::vector<Mat> >().swap(deltad2);
    bernoulli.clear();
    std::vector<std::vector<Mat> >().swap(bernoulli);
}

