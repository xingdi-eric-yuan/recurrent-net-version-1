#include "cost_gradient.h"

using namespace cv;
using namespace std;
void 
getNetworkCost(std::vector<Mat> &x, Mat &y, std::vector<Hl> &hLayers, Smr &smr){

    int T = x.size();
    int nSamples = x[0].cols;
    Mat tmp, tmp2;
    // hidden layer forward
    std::vector<std::vector<Mat> > nonlin_l;
    std::vector<std::vector<Mat> > acti_l;
    std::vector<std::vector<Mat> > bernoulli_l;

    std::vector<std::vector<Mat> > nonlin_r;
    std::vector<std::vector<Mat> > acti_r;
    std::vector<std::vector<Mat> > bernoulli_r;

    std::vector<Mat> tmp_vec;
    acti_l.push_back(tmp_vec);
    acti_r.push_back(tmp_vec); 
    for(int i = 0; i < T; ++i){
        acti_r[0].push_back(x[i]);
        acti_l[0].push_back(x[i]);
        tmp_vec.push_back(tmp);
    }

    for(int i = 1; i <= hiddenConfig.size(); ++i){
        // for each hidden layer
        acti_l.push_back(tmp_vec);
        nonlin_l.push_back(tmp_vec);
        bernoulli_l.push_back(tmp_vec);
        acti_r.push_back(tmp_vec);
        nonlin_r.push_back(tmp_vec);
        bernoulli_r.push_back(tmp_vec);
        // from left to right
        for(int j = 0; j < T; ++j){
            // for each time slot
            Mat tmpacti = hLayers[i - 1].U_l * acti_l[i - 1][j];
            if(j > 0) tmpacti += hLayers[i - 1].W_l * acti_l[i][j - 1];
            if(i > 1) tmpacti += hLayers[i - 1].U_l * acti_r[i - 1][j];
            tmpacti.copyTo(nonlin_l[i - 1][j]);
            tmpacti = ReLU(tmpacti);
            if(hiddenConfig[i - 1].DropoutRate < 1.0){
                Mat bnl = getBernoulliMatrix(tmpacti.rows, tmpacti.cols, hiddenConfig[i - 1].DropoutRate);
                tmp = tmpacti.mul(bnl);
                tmp.copyTo(acti_l[i][j]);
                bnl.copyTo(bernoulli_l[i - 1][j]);
            }else tmpacti.copyTo(acti_l[i][j]);
        }
        // from right to left
        for(int j = T - 1; j >= 0; --j){
            // for each time slot
            Mat tmpacti = hLayers[i - 1].U_r * acti_r[i - 1][j];
            if(j < T - 1) tmpacti += hLayers[i - 1].W_r * acti_r[i][j + 1];
            if(i > 1) tmpacti += hLayers[i - 1].U_r * acti_l[i - 1][j];
            tmpacti.copyTo(nonlin_r[i - 1][j]);
            tmpacti = ReLU(tmpacti);
            if(hiddenConfig[i - 1].DropoutRate < 1.0){
                Mat bnl = getBernoulliMatrix(tmpacti.rows, tmpacti.cols, hiddenConfig[i - 1].DropoutRate);
                tmp = tmpacti.mul(bnl);
                tmp.copyTo(acti_r[i][j]);
                bnl.copyTo(bernoulli_r[i - 1][j]);
            }else tmpacti.copyTo(acti_r[i][j]);
        }
    }
    // softmax layer forward
    std::vector<Mat> p;
    for(int i = 0; i < T; ++i){
        Mat M = smr.W_l * acti_l[acti_l.size() - 1][i];
        M += smr.W_r * acti_r[acti_r.size() - 1][i];
        M -= repeat(reduce(M, 0, CV_REDUCE_MAX), M.rows, 1);
        M = exp(M);
        Mat tmpp = divide(M, repeat(reduce(M, 0, CV_REDUCE_SUM), M.rows, 1));
        p.push_back(tmpp);
    }
    std::vector<Mat> groundTruth;
    for(int i = 0; i < T; ++i){
        Mat tmpgroundTruth = Mat::zeros(softmaxConfig.NumClasses, nSamples, CV_64FC1);
        for(int j = 0; j < nSamples; j++){
            tmpgroundTruth.ATD(y.ATD(i, j), j) = 1.0;
        }
        groundTruth.push_back(tmpgroundTruth);
    }
    double J1 = 0.0;
    for(int i = 0; i < T; i++){
        J1 +=  - sum1(groundTruth[i].mul(log(p[i])));
    }
    J1 /= nSamples;
    double J2 = (sum1(pow(smr.W_l, 2.0)) + sum1(pow(smr.W_r, 2.0))) * softmaxConfig.WeightDecay / 2;
    double J3 = 0.0; 
    double J4 = 0.0;
    for(int hl = 0; hl < hLayers.size(); hl++){
        J3 += (sum1(pow(hLayers[hl].W_l, 2.0)) + sum1(pow(hLayers[hl].W_r, 2.0))) * hiddenConfig[hl].WeightDecay / 2;
    }
    for(int hl = 0; hl < hLayers.size(); hl++){
        J4 += (sum1(pow(hLayers[hl].U_l, 2.0)) + sum1(pow(hLayers[hl].U_r, 2.0))) * hiddenConfig[hl].WeightDecay / 2;
    }
    smr.cost = J1 + J2 + J3 + J4;
    if(!is_gradient_checking) 
        cout<<", J1 = "<<J1<<", J2 = "<<J2<<", J3 = "<<J3<<", J4 = "<<J4<<", Cost = "<<smr.cost<<endl;

    // softmax layer backward
    tmp = - (groundTruth[0] - p[0]) * acti_l[acti_l.size() - 1][0].t();
    for(int i = 1; i < T; ++i){
        tmp += - (groundTruth[i] - p[i]) * acti_l[acti_l.size() - 1][i].t();
    }
    smr.W_lgrad =  tmp / nSamples + softmaxConfig.WeightDecay * smr.W_l;
    tmp = pow((groundTruth[0] - p[0]), 2.0) * pow(acti_l[acti_l.size() - 1][0].t(), 2.0);
    for(int i = 1; i < T; ++i){
        tmp += pow((groundTruth[i] - p[i]), 2.0) * pow(acti_l[acti_l.size() - 1][i].t(), 2.0);
    }
    smr.W_ld2 = tmp / nSamples + softmaxConfig.WeightDecay;

    tmp = - (groundTruth[0] - p[0]) * acti_r[acti_r.size() - 1][0].t();
    for(int i = 1; i < T; ++i){
        tmp += - (groundTruth[i] - p[i]) * acti_r[acti_r.size() - 1][i].t();
    }
    smr.W_rgrad =  tmp / nSamples + softmaxConfig.WeightDecay * smr.W_r;
    tmp = pow((groundTruth[0] - p[0]), 2.0) * pow(acti_r[acti_r.size() - 1][0].t(), 2.0);
    for(int i = 1; i < T; ++i){
        tmp += pow((groundTruth[i] - p[i]), 2.0) * pow(acti_r[acti_r.size() - 1][i].t(), 2.0);
    }
    smr.W_rd2 = tmp / nSamples + softmaxConfig.WeightDecay;

    // hidden layer backward
    std::vector<std::vector<Mat> > delta_l(acti_l.size());
    std::vector<std::vector<Mat> > delta_ld2(acti_l.size());
    std::vector<std::vector<Mat> > delta_r(acti_r.size());
    std::vector<std::vector<Mat> > delta_rd2(acti_r.size());
    for(int i = 0; i < delta_l.size(); i++){
        delta_l[i].clear();
        delta_ld2[i].clear();
        delta_r[i].clear();
        delta_rd2[i].clear();
        Mat tmpmat;
        for(int j = 0; j < T; j++){
            delta_l[i].push_back(tmpmat);
            delta_ld2[i].push_back(tmpmat);
            delta_r[i].push_back(tmpmat);
            delta_rd2[i].push_back(tmpmat);
        }
    }
    // Last hidden layer
    // Do BPTT backward pass for the forward hidden layer
    for(int i = T - 1; i >= 0; i--){
        tmp = -smr.W_l.t() * (groundTruth[i] - p[i]);
        tmp2 = pow(smr.W_l.t(), 2.0) * pow((groundTruth[i] - p[i]), 2.0);
        if(i < T - 1){
            tmp += hLayers[hLayers.size() - 1].W_l.t() * delta_l[delta_l.size() - 1][i + 1];
            tmp2 += pow(hLayers[hLayers.size() - 1].W_l.t(), 2.0) * delta_ld2[delta_ld2.size() - 1][i + 1];
        }
        tmp.copyTo(delta_l[delta_l.size() - 1][i]);
        tmp2.copyTo(delta_ld2[delta_ld2.size() - 1][i]);
        delta_l[delta_l.size() - 1][i] = delta_l[delta_l.size() - 1][i].mul(dReLU(nonlin_l[nonlin_l.size() - 1][i]));
        delta_ld2[delta_ld2.size() - 1][i] = delta_ld2[delta_ld2.size() - 1][i].mul(pow(dReLU(nonlin_l[nonlin_l.size() - 1][i]), 2.0));
        if(hiddenConfig[hiddenConfig.size() - 1].DropoutRate < 1.0){
            delta_l[delta_l.size() - 1][i] = delta_l[delta_l.size() -1][i].mul(bernoulli_l[bernoulli_l.size() - 1][i]);
            delta_ld2[delta_ld2.size() - 1][i] = delta_ld2[delta_ld2.size() -1][i].mul(pow(bernoulli_l[bernoulli_l.size() - 1][i], 2.0));
        } 
    }
    // Do BPTT backward pass for the backward hidden layer
    for(int i = 0; i < T; i++){
        tmp = -smr.W_r.t() * (groundTruth[i] - p[i]);
        tmp2 = pow(smr.W_r.t(), 2.0) * pow((groundTruth[i] - p[i]), 2.0);
        if(i > 0){
            tmp += hLayers[hLayers.size() - 1].W_r.t() * delta_r[delta_r.size() - 1][i - 1];
            tmp2 += pow(hLayers[hLayers.size() - 1].W_r.t(), 2.0) * delta_rd2[delta_rd2.size() - 1][i - 1];
        }
        tmp.copyTo(delta_r[delta_r.size() - 1][i]);
        tmp2.copyTo(delta_rd2[delta_rd2.size() - 1][i]);
        delta_r[delta_r.size() - 1][i] = delta_r[delta_r.size() - 1][i].mul(dReLU(nonlin_r[nonlin_r.size() - 1][i]));
        delta_rd2[delta_rd2.size() - 1][i] = delta_rd2[delta_rd2.size() - 1][i].mul(pow(dReLU(nonlin_r[nonlin_r.size() - 1][i]), 2.0));
        if(hiddenConfig[hiddenConfig.size() - 1].DropoutRate < 1.0){
            delta_r[delta_r.size() - 1][i] = delta_r[delta_r.size() -1][i].mul(bernoulli_r[bernoulli_r.size() - 1][i]);
            delta_rd2[delta_rd2.size() - 1][i] = delta_rd2[delta_rd2.size() -1][i].mul(pow(bernoulli_r[bernoulli_r.size() - 1][i], 2.0));
        } 
    }
    // hidden layers
    for(int i = delta_l.size() - 2; i > 0; --i){
        // Do BPTT backward pass for the forward hidden layer
        for(int j = T - 1; j >= 0; --j){
            tmp = hLayers[i].U_l.t() * delta_l[i + 1][j];
            tmp2 = pow(hLayers[i].U_l.t(), 2.0) * delta_ld2[i + 1][j];
            if(j < T - 1){
                tmp += hLayers[i - 1].W_l.t() * delta_l[i][j + 1];
                tmp2 += pow(hLayers[i - 1].W_l.t(), 2.0) * delta_ld2[i][j + 1];
            }
            tmp += hLayers[i].U_r.t() * delta_r[i + 1][j];
            tmp2 += pow(hLayers[i].U_r.t(), 2.0) * delta_rd2[i + 1][j];
            tmp.copyTo(delta_l[i][j]);
            tmp2.copyTo(delta_ld2[i][j]);
            delta_l[i][j] = delta_l[i][j].mul(dReLU(nonlin_l[i - 1][j]));
            delta_ld2[i][j] = delta_ld2[i][j].mul(pow(dReLU(nonlin_l[i - 1][j]), 2.0));
            if(hiddenConfig[i - 1].DropoutRate < 1.0){
                delta_l[i][j] = delta_l[i][j].mul(bernoulli_l[i - 1][j]);
                delta_ld2[i][j] = delta_ld2[i][j].mul(pow(bernoulli_l[i - 1][j], 2.0));
            }
        }
        // Do BPTT backward pass for the backward hidden layer
        for(int j = 0; j < T; ++j){
            tmp = hLayers[i].U_r.t() * delta_r[i + 1][j];
            tmp2 = pow(hLayers[i].U_r.t(), 2.0) * delta_rd2[i + 1][j];
            if(j > 0){
                tmp += hLayers[i - 1].W_r.t() * delta_r[i][j - 1];
                tmp2 += pow(hLayers[i - 1].W_r.t(), 2.0) * delta_rd2[i][j - 1];
            }
            tmp += hLayers[i].U_l.t() * delta_l[i + 1][j];
            tmp2 += pow(hLayers[i].U_l.t(), 2.0) * delta_ld2[i + 1][j];
            tmp.copyTo(delta_r[i][j]);
            tmp2.copyTo(delta_rd2[i][j]);
            delta_r[i][j] = delta_r[i][j].mul(dReLU(nonlin_r[i - 1][j]));
            delta_rd2[i][j] = delta_rd2[i][j].mul(pow(dReLU(nonlin_r[i - 1][j]), 2.0));
            if(hiddenConfig[i - 1].DropoutRate < 1.0){
                delta_r[i][j] = delta_r[i][j].mul(bernoulli_r[i - 1][j]);
                delta_rd2[i][j] = delta_rd2[i][j].mul(pow(bernoulli_r[i - 1][j], 2.0));
            }
        }
    }

    for(int i = hiddenConfig.size() - 1; i >= 0; i--){
        // forward part.
        if(i == 0){
            tmp = delta_l[i + 1][0] * acti_l[i][0].t();
            tmp2 = delta_ld2[i + 1][0] * pow(acti_l[i][0].t(), 2.0);
            for(int j = 1; j < T; ++j){
                tmp += delta_l[i + 1][j] * acti_l[i][j].t();
                tmp2 += delta_ld2[i + 1][j] * pow(acti_l[i][j].t(), 2.0);
            }
        }else{
            tmp = delta_l[i + 1][0] * (acti_l[i][0].t() + acti_r[i][0].t());
            tmp2 = delta_ld2[i + 1][0] * (pow(acti_l[i][0].t(), 2.0) + pow(acti_r[i][0].t(), 2.0));
            for(int j = 1; j < T; ++j){
                tmp += delta_l[i + 1][j] * (acti_l[i][j].t() + acti_r[i][j].t());
                tmp2 += delta_ld2[i + 1][j] * (pow(acti_l[i][j].t(), 2.0) + pow(acti_r[i][j].t(), 2.0));
            }
        }
        hLayers[i].U_lgrad = tmp / nSamples + hiddenConfig[i].WeightDecay * hLayers[i].U_l;
        hLayers[i].U_ld2 = tmp2 / nSamples + hiddenConfig[i].WeightDecay;

        tmp = delta_l[i + 1][T - 1] * acti_l[i + 1][T - 2].t();
        tmp2 = delta_ld2[i + 1][T - 1] * pow(acti_l[i + 1][T - 2].t(), 2.0);
        for(int j = T - 2; j > 0; j--){
            tmp += delta_l[i + 1][j] * acti_l[i + 1][j - 1].t();
            tmp2 += delta_ld2[i + 1][j] * pow(acti_l[i + 1][j - 1].t(), 2.0);
        }
        hLayers[i].W_lgrad = tmp / nSamples + hiddenConfig[i].WeightDecay * hLayers[i].W_l;
        hLayers[i].W_ld2 = tmp2 / nSamples + hiddenConfig[i].WeightDecay;

        // backward part.
        if(i == 0){
            tmp = delta_r[i + 1][0] * acti_r[i][0].t();
            tmp2 = delta_rd2[i + 1][0] * pow(acti_r[i][0].t(), 2.0);
            for(int j = 1; j < T; ++j){
                tmp += delta_r[i + 1][j] * acti_r[i][j].t();
                tmp2 += delta_rd2[i + 1][j] * pow(acti_r[i][j].t(), 2.0);
            }
        }else{
            tmp = delta_r[i + 1][0] * (acti_l[i][0].t() + acti_r[i][0].t());
            tmp2 = delta_rd2[i + 1][0] * (pow(acti_l[i][0].t(), 2.0) + pow(acti_r[i][0].t(), 2.0));
            for(int j = 1; j < T; ++j){
                tmp += delta_r[i + 1][j] * (acti_l[i][j].t() + acti_r[i][j].t());
                tmp2 += delta_rd2[i + 1][j] * (pow(acti_l[i][j].t(), 2.0) + pow(acti_r[i][j].t(), 2.0));
            }
        }
        hLayers[i].U_rgrad = tmp / nSamples + hiddenConfig[i].WeightDecay * hLayers[i].U_r;
        hLayers[i].U_rd2 = tmp2 / nSamples + hiddenConfig[i].WeightDecay;

        tmp = delta_r[i + 1][0] * acti_r[i + 1][1].t();
        tmp2 = delta_rd2[i + 1][0] * pow(acti_r[i + 1][1].t(), 2.0);
        for(int j = 1; j < T - 1; j++){
            tmp += delta_r[i + 1][j] * acti_r[i + 1][j + 1].t();
            tmp2 += delta_rd2[i + 1][j] * pow(acti_r[i + 1][j + 1].t(), 2.0);
        }
        hLayers[i].W_rgrad = tmp / nSamples + hiddenConfig[i].WeightDecay * hLayers[i].W_r;
        hLayers[i].W_rd2 = tmp2 / nSamples + hiddenConfig[i].WeightDecay;
    }
    // destructor
    acti_l.clear();
    std::vector<std::vector<Mat> >().swap(acti_l);
    nonlin_l.clear();
    std::vector<std::vector<Mat> >().swap(nonlin_l);
    delta_l.clear();
    std::vector<std::vector<Mat> >().swap(delta_l);
    delta_ld2.clear();
    std::vector<std::vector<Mat> >().swap(delta_ld2);
    bernoulli_l.clear();
    std::vector<std::vector<Mat> >().swap(bernoulli_l);
    acti_r.clear();
    std::vector<std::vector<Mat> >().swap(acti_r);
    nonlin_r.clear();
    std::vector<std::vector<Mat> >().swap(nonlin_r);
    delta_r.clear();
    std::vector<std::vector<Mat> >().swap(delta_r);
    delta_rd2.clear();
    std::vector<std::vector<Mat> >().swap(delta_rd2);
    bernoulli_r.clear();
    std::vector<std::vector<Mat> >().swap(bernoulli_r);
    p.clear();
    std::vector<Mat>().swap(p);
    groundTruth.clear();
    std::vector<Mat>().swap(groundTruth);
}

