#pragma once
#include "general_settings.h"
#include <sys/types.h>
#include <sys/stat.h>

using namespace std;
using namespace cv;

void save2txt(const Mat &, string, string);

//void save2XML(string, string, const std::vector<Cvl> &, const std::vector<Fcl> &, const Smr &, const std::vector<string>&);

//void readFromXML(string , std::vector<Cvl> &, std::vector<Fcl> &, Smr &, std::vector<string> &);