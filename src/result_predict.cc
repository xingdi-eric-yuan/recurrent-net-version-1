#include "result_predict.h"

using namespace cv;
using namespace std;

Mat 
resultPredict(std::vector<Mat> &x, std::vector<Hl> &hLayers, Smr &smr){

    int T = x.size();
    int mid = (int)(T /2.0);
    Mat tmp;
    // hidden layer forward
    std::vector<std::vector<Mat> > acti_l;
    std::vector<std::vector<Mat> > acti_r;
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
        acti_r.push_back(tmp_vec);
        // from left to right
        for(int j = 0; j < T; ++j){
            // for each time slot
            Mat tmpacti = hLayers[i - 1].U_l * acti_l[i - 1][j];
            if(j > 0) tmpacti += hLayers[i - 1].W_l * acti_l[i][j - 1];
            if(i > 1) tmpacti += hLayers[i - 1].U_l * acti_r[i - 1][j];
            tmpacti = ReLU(tmpacti);
            if(hiddenConfig[i - 1].DropoutRate < 1.0) tmpacti = tmpacti.mul(hiddenConfig[i - 1].DropoutRate);
            tmpacti.copyTo(acti_l[i][j]);
        }
        // from right to left
        for(int j = T - 1; j >= 0; --j){
            // for each time slot
            Mat tmpacti = hLayers[i - 1].U_r * acti_r[i - 1][j];
            if(j < T - 1) tmpacti += hLayers[i - 1].W_r * acti_r[i][j + 1];
            if(i > 1) tmpacti += hLayers[i - 1].U_r * acti_l[i - 1][j];
            tmpacti = ReLU(tmpacti);
            if(hiddenConfig[i - 1].DropoutRate < 1.0) tmpacti = tmpacti.mul(hiddenConfig[i - 1].DropoutRate);
            tmpacti.copyTo(acti_r[i][j]);
        }
    }
    tmp_vec.clear();
    std::vector<Mat>().swap(tmp_vec);
    // softmax layer forward
    Mat M = smr.W_l * acti_l[acti_l.size() - 1][mid];
    M += smr.W_r * acti_r[acti_r.size() - 1][mid];
    Mat result = Mat::zeros(1, M.cols, CV_64FC1);

    double minValue, maxValue;
    Point minLoc, maxLoc;
    for(int i = 0; i < M.cols; i++){
        minMaxLoc(M(Rect(i, 0, 1, M.rows)), &minValue, &maxValue, &minLoc, &maxLoc);
        result.ATD(0, i) = (int)maxLoc.y;
    }
    acti_l.clear();
    std::vector<std::vector<Mat> >().swap(acti_l);
    acti_r.clear();
    std::vector<std::vector<Mat> >().swap(acti_r);
    return result;
}

void 
testNetwork(const std::vector<std::vector<int> > &x, std::vector<std::vector<int> > &y, std::vector<Hl> &HiddenLayers, Smr &smr, 
             std::vector<string> &re_wordmap){

    // Test use test set
    // Because it may leads to lack of memory if testing the whole dataset at 
    // one time, so separate the dataset into small pieces of batches (say, batch size = 20).
    // 
    int batchSize = 50;
    Mat result = Mat::zeros(1, x.size(), CV_64FC1);

    std::vector<std::vector<int> > tmpBatch;
    int batch_amount = x.size() / batchSize;
    for(int i = 0; i < batch_amount; i++){
        for(int j = 0; j < batchSize; j++){
            tmpBatch.push_back(x[i * batchSize + j]);
        }
        std::vector<Mat> sampleX;
        getDataMat(tmpBatch, sampleX, re_wordmap);
        Mat resultBatch = resultPredict(sampleX, HiddenLayers, smr);
        Rect roi = Rect(i * batchSize, 0, batchSize, 1);
        resultBatch.copyTo(result(roi));
        tmpBatch.clear();
        sampleX.clear();
    }
    if(x.size() % batchSize){
        for(int j = 0; j < x.size() % batchSize; j++){
            tmpBatch.push_back(x[batch_amount * batchSize + j]);
        }
        std::vector<Mat> sampleX;
        getDataMat(tmpBatch, sampleX, re_wordmap);
        Mat resultBatch = resultPredict(sampleX, HiddenLayers, smr);
        Rect roi = Rect(batch_amount * batchSize, 0, x.size() % batchSize, 1);
        resultBatch.copyTo(result(roi));
        ++ batch_amount;
        tmpBatch.clear();
        sampleX.clear();
    }
    Mat sampleY = Mat::zeros(1, y.size(), CV_64FC1);
    getLabelMat(y, sampleY);

    Mat err;
    sampleY.copyTo(err);
    err -= result;
    int correct = err.cols;
    for(int i=0; i<err.cols; i++){
        if(err.ATD(0, i) != 0) --correct;
    }
    cout<<"######################################"<<endl;
    cout<<"## test result. "<<correct<<" correct of "<<err.cols<<" total."<<endl;
    cout<<"## Accuracy is "<<(double)correct / (double)(err.cols)<<endl;
    cout<<"######################################"<<endl;
    result.release();
    err.release();
}

