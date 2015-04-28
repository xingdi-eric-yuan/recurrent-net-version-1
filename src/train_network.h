#pragma once
#include "general_settings.h"

using namespace std;
using namespace cv;

void trainNetwork(const std::vector<std::vector<int> > &, std::vector<std::vector<int> > &, std::vector<Hl> &, Smr &, 
				  const std::vector<std::vector<int> > &, std::vector<std::vector<int> > &, std::vector<string>&);