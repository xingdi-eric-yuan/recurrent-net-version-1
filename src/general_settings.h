#pragma once
#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "data_structure.h"
#include "read_data.h"
#include "helper.h"
#include "cost_gradient.h"
#include "gradient_checking.h"
#include "helper.h"
#include "matrix_maths.h"
#include "weights_IO.h"
#include "train_network.h"
#include "weight_init.h"
#include "result_predict.h"
#include "read_config.h"

// non-linearity
#define NL_SIGMOID 0
#define NL_TANH 1
#define NL_RELU 2

#define ATD at<double>
#define elif else if

using namespace std;
using namespace cv;

///////////////////////////////////
// General parameters
///////////////////////////////////
extern float training_percent;


extern std::vector<HiddenLayerConfig> hiddenConfig;
extern SoftmaxLayerConfig softmaxConfig;
extern std::vector<int> sample_vec;

///////////////////////////////////
// General parameters
///////////////////////////////////
extern bool is_gradient_checking;
extern bool use_log;
extern int batch_size;
extern int log_iter;

extern int non_linearity;
extern int training_epochs;
extern double lrate_w;
extern double lrate_b;
extern int iter_per_epo;
extern int nGram;
extern int word_vec_len;

