#define CNN
#define quantize 0
#define debug 1
#include<iostream>
#include<fstream>
#include<vector>
#include<string>
#include<cstdio>
#include<sstream>
#include<cmath>
#include<opencv2/opencv.hpp>
using namespace std;
using namespace cv;

typedef vector<vector<vector<float> > > image_type; // image[channel number][row][column]

typedef struct KERNEL{
    vector<vector<vector<vector<float> > > > kernel; // kernel[output_channel][input_channel][row][col]
    int output_size, input_size, kernel_size;
}kernel_type;

typedef struct BIAS{
    vector<float> bias;
    int size;
}bias_type;

#if (quantize == 0)
typedef struct Statistics{
    vector<float> running_mean;
    vector<float> running_variance;
    int size;
}statistic_type;

typedef struct BN{
    vector<float> bn_w;
    vector<float> bn_b;
    int size;
}bn_type;
#endif

typedef struct fc_W{
    vector<vector<float> > weight;
    int output_size, input_size;
}fc_weight;
typedef struct fc_B{
    vector<float> bias;
    int size;
}fc_bias;
typedef vector<float> fc_type;

class cnn{
    public:
        cnn();
        ~cnn();
        void r_img(string filename, string im_type);
        image_type resize(int newsize, int newchannel);
        #if (quantize == 1)
        void conv(kernel_type kernel, bias_type bias, int stride);
        #elif (quantize == 0)
        void conv(kernel_type kernel, statistic_type statistic, bn_type bn, int stride);
        #endif
        image_type padding(image_type before_padding, int kernel_size, int stride);
        void ReLu();
        void maxpooling(int kernel_size, int stride);
        void avgpooling();
        void t(int s);
        image_type get_channel();
    private:
        image_type channel;
        int size;
        int channel_num;
};

kernel_type get_kernel(string kernel_file);
bias_type get_bias(string bias_file);

#if (quantize == 0)
statistic_type get_stat(string run_m, string run_v);
bn_type get_bn(string weight, string bias);
#endif

fc_weight get_fc_weight(string fc_file_w);
fc_bias get_fc_bias(string fc_file_b);
fc_type fc(image_type rgb, image_type depth, fc_weight fc_w, fc_bias fc_b);
fc_type full_connected(image_type rgb, image_type depth, string weight_file_name, string bias_file_name, int out_size);
string ToString(int sel);