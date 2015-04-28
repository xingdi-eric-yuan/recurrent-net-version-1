#pragma once
#include "general_settings.h"

using namespace cv;
using namespace std;

void weightRandomInit(Hl&, int, int);

void weightRandomInit(Smr&, int, int);

void rnnInitPrarms(std::vector<Hl>&, Smr&);