#define CNN
#define quantize 1  // 0 for 32*32, 1 for 32*8 , 2 for 8*8
#define debug 0
#include<iostream>
#include<fstream>
#include<vector>
#include<string>
#include<cstdio>
#include<sstream>
#include<cmath>
#include<tuple>
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
        void conv(kernel_type kernel, bias_type running_mean, bias_type running_var, bias_type bn_w, bias_type bn_b, int stride);
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

fc_weight get_fc_weight(string fc_file_w);
fc_bias get_fc_bias(string fc_file_b);
fc_type fc(image_type rgb, image_type depth, fc_weight fc_w, fc_bias fc_b);
fc_type full_connected(image_type rgb, image_type depth, string weight_file_name, string bias_file_name, int out_size);
string ToString(int sel);