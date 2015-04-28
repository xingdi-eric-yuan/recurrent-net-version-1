#include "weights_IO.h"

#define AT3D at<cv::Vec3d>
using namespace cv;
using namespace std;

void 
save2txt(const Mat &data, string path, string str){
    string tmp = path + str;
    FILE *pOut = fopen(tmp.c_str(), "w");
    for(int i = 0; i < data.rows; i++){
        for(int j = 0; j < data.cols; j++){
            fprintf(pOut, "%lf", data.ATD(i, j));
            if(j == data.cols - 1){
                fprintf(pOut, "\n");
            } 
            else{
                fprintf(pOut, " ");
            } 
        }
    }
    fclose(pOut);
}

/*
void 
save2XML(string path, string name, const std::vector<Cvl> &CLayers, const std::vector<Fcl> &FClayers, const Smr &smr, const std::vector<string> &re_resolmap){

    mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    string tmp = path + "/" + name + ".xml";
    FileStorage fs(tmp, FileStorage::WRITE);
    
    fs << "smr_W" << smr.W;
    fs << "smr_b" << smr.b;
    for(int i = 0; i < CLayers.size(); i++){
        for(int j = 0; j < convConfig[i].KernelAmount; j++){
            tmp = "convlayer" + std::to_string(i) + "_kernel_" + std::to_string(j);
            fs << (tmp + "_W") << CLayers[i].layer[j].W;
            fs << (tmp + "_b") << CLayers[i].layer[j].b;
        }
    }
    for(int i = 0; i < FClayers.size(); i++){
        tmp = "fclayer" + std::to_string(i);
        fs << (tmp + "_W") << FClayers[i].W;
        fs << (tmp + "_b") << FClayers[i].b;
    }
    fs << "re_resolmap" << re_resolmap;
    fs.release();
    cout<<"Successfully saved network information..."<<endl;
}

void 
readFromXML(string path, std::vector<Cvl> &CLayers, std::vector<Fcl> &FClayers, Smr &smr, std::vector<string> &re_resolmap){

    string tmp = "";
    FileStorage fs(path, FileStorage::READ);
    fs["smr_W"] >> smr.W;
    fs["smr_b"] >> smr.b;
    for(int i = 0; i < CLayers.size(); i++){
        for(int j = 0; j < convConfig[i].KernelAmount; j++){
            tmp = "convlayer" + std::to_string(i) + "_kernel_" + std::to_string(j);
            fs[tmp + "_W"] >> CLayers[i].layer[j].W;
            fs[tmp + "_b"] >> CLayers[i].layer[j].b;
        }
    }
    for(int i = 0; i < FClayers.size(); i++){
        tmp = "fclayer" + std::to_string(i);
        fs[tmp + "_W"] >> FClayers[i].W;
        fs[tmp + "_b"] >> FClayers[i].b;
    }
    fs["re_resolmap"] >> re_resolmap;
    fs.release();
    cout<<"Successfully read network information from "<<path<<"..."<<endl;
}
*/
