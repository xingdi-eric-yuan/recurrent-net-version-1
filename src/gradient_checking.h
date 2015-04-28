#pragma once
#include "general_settings.h"

using namespace std;
using namespace cv;

void gradientChecking_SoftmaxLayer(std::vector<Hl> &, Smr &, std::vector<Mat> &, Mat&);
void gradientChecking_HiddenLayer (std::vector<Hl> &, Smr &, std::vector<Mat> &, Mat&, int);
