#pragma once
#include "general_settings.h"

using namespace std;
using namespace cv;

Mat resultPredict(std::vector<Mat> &, std::vector<Hl> &, Smr &);

void testNetwork(const std::vector<std::vector<int> > &, std::vector<std::vector<int> > &, std::vector<Hl> &, Smr &, std::vector<string> &);