#include"cnn.hpp"

cnn::cnn():channel(0), size(256), channel_num(3){}
cnn::~cnn(){}
#if (quantize == 1)
void cnn::r_img(string filename, string im_type){
    Mat image;
    if(im_type == "RGB"){
        image = imread(filename, IMREAD_COLOR);
        if(image.empty()){   // Check for invalid input
            cout << "Could not open or find the image" << endl;
            exit(-1);
        }
        channel = cnn::resize(size, channel_num);
        for(int i=0;i<image.rows;i++){
            for(int j=0;j<image.cols;j++){
                Vec3b rgbPixel;
                rgbPixel = image.at<Vec3b>(i, j);//BGR
                for(int ch=0;ch<3;ch++){
                    channel[ch][i][j] = rgbPixel.val[2-ch];
                    if(channel[ch][i][j] > 127.0)
                        channel[ch][i][j] = 127.0;
                    channel[ch][i][j] = (channel[ch][i][j]/128.0) - 0.5;
                }
            }
        }
    }else if(im_type == "Depth"){
        image = imread(filename, IMREAD_GRAYSCALE);
        if(image.empty()){
            cout << "Could not open or find the image" << endl;
            exit(-1);
        }
        channel_num = 1;
        channel = cnn::resize(size, channel_num);
        for(int i=0;i<image.rows;i++){
            for(int j=0;j<image.cols;j++){
                uchar depthPixel;
                depthPixel = image.at<uchar>(i,j);
                channel[0][i][j] = depthPixel;
                if(channel[0][i][j] > 127.0)
                    channel[0][i][j] = 127.0;
                channel[0][i][j] = (channel[0][i][j]/128.0) - 0.5;
            }
        }
    }
    //cout << channel.size() << endl;
    //cout << channel[0].size() << endl;
    //cout << channel[0][0].size() << endl;
}
#elif (quantize == 0)
void cnn::r_img(string filename, string im_type){
    Mat image;
    if(im_type == "RGB"){
        image = imread(filename, IMREAD_COLOR);
        if(image.empty()){   // Check for invalid input
            cout << "Could not open or find the image" << endl;
            exit(-1);
        }
        channel = cnn::resize(size, channel_num);
        for(int i=0;i<image.rows;i++){
            for(int j=0;j<image.cols;j++){
                Vec3b rgbPixel;
                rgbPixel = image.at<Vec3b>(i, j);//BGR
                for(int ch=0;ch<3;ch++){
                    channel[ch][i][j] = rgbPixel.val[2-ch]/256.0;
                    /*if(channel[ch][i][j] > 127.0)
                        channel[ch][i][j] = 127.0;
                    channel[ch][i][j] = (channel[ch][i][j]/128.0) - 0.5;*/
                }
            }
        }
        #if (debug)
        for(int i=0;i<image.rows;i++){
            for(int j=0;j<image.cols;j++){
                for(int ch=0;ch<3;ch++){
                    channel[ch][i][j] = 1.0;
                }
            }
        }
        #endif
    }else if(im_type == "Depth"){
        image = imread(filename, IMREAD_GRAYSCALE);
        if(image.empty()){
            cout << "Could not open or find the image" << endl;
            exit(-1);
        }
        channel_num = 1;
        channel = cnn::resize(size, channel_num);
        for(int i=0;i<image.rows;i++){
            for(int j=0;j<image.cols;j++){
                uchar depthPixel;
                depthPixel = image.at<uchar>(i,j);
                channel[0][i][j] = depthPixel/256.0;
                /*if(channel[0][i][j] > 127.0)
                    channel[0][i][j] = 127.0;
                channel[0][i][j] = (channel[0][i][j]/128.0) - 0.5;*/
            }
        }
        #if (debug)
        for(int i=0;i<image.rows;i++){
            for(int j=0;j<image.cols;j++){
                channel[0][i][j] = 1.0;
            }
        }
        #endif
    }
    //cout << channel.size() << endl;
    //cout << channel[0].size() << endl;
    //cout << channel[0][0].size() << endl;
}
#endif

image_type cnn::resize(int newsize, int newchannel_num){ // image[channel][row][col]
    image_type im;
    im.clear();
    im.resize(newchannel_num);
    for(int i=0;i<im.size();i++){
        im[i].resize(newsize);
        for(int j=0;j<im[i].size();j++){
            im[i][j].resize(newsize);
        }
    }
    return im;
}

#if (quantize == 1)
void cnn::conv(kernel_type kernel, bias_type bias, int stride){     // kernel[output_channel][input_channel][row][col]
    image_type padding_channel;                                     // padding_channel[channel_num][row][col]
    padding_channel = padding(channel, kernel.kernel_size, stride);
    channel = resize(size, kernel.output_size);

    for(int ch_o=0;ch_o<kernel.output_size;ch_o++){
        for(int row=0;row<size;row++){
            for(int col=0;col<size;col++){
                for(int ch_i=0;ch_i<kernel.input_size;ch_i++){
                    for(int p=0;p<kernel.kernel_size;p++){
                        for(int q=0;q<kernel.kernel_size;q++){
                            channel[ch_o][row][col] += padding_channel[ch_i][row*stride+p][col*stride+q]*kernel.kernel[ch_o][ch_i][p][q];
                        }
                    }
                }
                channel[ch_o][row][col] += bias.bias[ch_o];
            }
        }
    }

    ReLu();
    channel_num = channel.size();
    /*
    for(int i=0;i<4;i++){
        for(int j=0;j<4;j++){
            cout << channel[0][i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;*/
}
#elif (quantize == 0)
void cnn::conv(kernel_type kernel, statistic_type statistic, bn_type bn, int stride){     // kernel[output_channel][input_channel][row][col]
    image_type padding_channel;                     // padding_channel[channel_num][row][col]
    padding_channel = padding(channel, kernel.kernel_size, stride);
    channel = resize(size, kernel.output_size);

    for(int ch_o=0;ch_o<kernel.output_size;ch_o++){
        for(int row=0;row<size;row++){
            for(int col=0;col<size;col++){
                for(int ch_i=0;ch_i<kernel.input_size;ch_i++){
                    for(int p=0;p<kernel.kernel_size;p++){
                        for(int q=0;q<kernel.kernel_size;q++){
                            channel[ch_o][row][col] += padding_channel[ch_i][row*stride+p][col*stride+q]*kernel.kernel[ch_o][ch_i][p][q];
                        }
                    }
                }
                channel[ch_o][row][col] = ((channel[ch_o][row][col]-statistic.running_mean[ch_o])/sqrt(statistic.running_variance[ch_o]))*bn.bn_w[ch_o]+bn.bn_b[ch_o];
            }
        }
    }
    #if (debug)
    for(int i=0;i<5;i++){
        for(int j=0;j<5;j++){
            cout << channel[5][i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;
    #endif
    ReLu();
    channel_num = channel.size();
}
#endif

image_type cnn::padding(image_type before_padding, int kernel_size, int stride){
    image_type after_padding;
    int new_size, pad;
    pad = ((before_padding[0].size()-1)*stride-before_padding[0].size()+kernel_size)/2;
    new_size = before_padding[0].size()+2*pad;
    after_padding = resize(new_size, before_padding.size());

    for(int k=0;k<before_padding.size();k++){
        for(int i=0;i<before_padding[0].size();i++){
            for(int j=0;j<before_padding[0][0].size();j++){
                after_padding[k][i+pad][j+pad] = before_padding[k][i][j];
            }
        }
    }
    return after_padding;
}

void cnn::ReLu(){
    for(int i=0;i<channel.size();i++){
        for(int j=0;j<channel[0].size();j++){
            for(int k=0;k<channel[0][0].size();k++){
                if(channel[i][j][k] < 0)
                    channel[i][j][k] = 0;
            }
        }
    }
}

void cnn::maxpooling(int kernel_size, int stride){
    size = size/stride;
    image_type result;
    result = resize(size, channel_num);

    for(int cout=0;cout<channel_num;cout++){
        for(int row=0;row<size;row++){
            for(int col=0;col<size;col++){
                float temp = channel[cout][row*stride][col*stride];
                for(int p=0;p<kernel_size;p++){
                    for(int q=0;q<kernel_size;q++){
                        if(temp<channel[cout][row*stride+p][col*stride+q]){
                            temp = channel[cout][row*stride+p][col*stride+q];
                        }
                    }
                }
                result[cout][row][col] = temp;
            }
        }
    }
    channel.clear();
    channel = resize(size, channel_num);
    channel = result;
    /*for(int i=0;i<2;i++){
        for(int j=0;j<2;j++){
            cout << result[0][i][j] << " ";
        }
        cout << endl;
    }
    cout << result[0].size() << endl;*/
    cout << channel_num << " " << size << endl;
}

void cnn::avgpooling(){
    float tmp = 0;
    image_type result;
    result = resize(1, 64);
    //cout << "size: " << size << endl;
    for(int i=0;i<channel_num;i++){
        for(int row=0;row<size;row++){
            for(int col=0;col<size;col++){
                tmp += channel[i][row][col];
            }
        }
        tmp /= 64;
        result[i][0][0] = tmp;
        //cout << i << ": " << tmp << endl;
    }
    size = 1;
    channel.clear();
    channel = resize(1, 64);
    channel = result;
    cout << channel_num << " " << size << endl;
    //cout << channel[0][0][0] << endl;
}

kernel_type get_kernel(string kernel_file){
    kernel_type kernel;
    int input_size, output_size, kernel_size;
    ifstream fin(kernel_file);
    string data;
    getline(fin, data);
    //cout << data << endl;
    sscanf(data.c_str(), "# torch.Size([%d, %d, %d, %d])", &output_size, &input_size, &kernel_size, &kernel_size);
    kernel.output_size = output_size;
    kernel.input_size = input_size;
    kernel.kernel_size = kernel_size;

    kernel.kernel.resize(output_size);
    for(int i=0;i<output_size;i++){
        kernel.kernel[i].resize(input_size);
        for(int j=0;j<input_size;j++){
            kernel.kernel[i][j].resize(kernel_size);
            for(int k=0;k<kernel_size;k++){
                kernel.kernel[i][j][k].resize(kernel_size);
            }
        }
    }

    for(int i=0;i<output_size;i++){
        getline(fin, data);
        stringstream stream(data);
        for(int j=0;j<input_size;j++){
            for(int k=0;k<kernel_size;k++){
                for(int m=0;m<kernel_size;m++){
                    float temp;
                    stream >> temp;
                    kernel.kernel[i][j][k][m] = temp;
                }
            }
        }
    }
    /*for(int i=0;i<3;i++){
        for(int j=0;j<3;j++){
            cout << i << j << " " << kernel.kernel[0][1][i][j] << endl;
        }
    }*/
    return kernel;
}

bias_type get_bias(string bias_file){
    bias_type bias;
    string data;
    ifstream fin(bias_file);
    int size;
    getline(fin, data);
    sscanf(data.c_str(), "# torch.Size([%d])", &size);
    bias.size = size;

    bias.bias.resize(size);
    for(int i=0;i<size;i++){
        getline(fin, data);
        stringstream stream(data);
        float temp;
        stream >> temp;
        bias.bias[i] = temp;
    }
    //cout << bias.bias[0] << endl;
    //cout << bias.size << endl;
    return bias;
}

#if (quantize == 0)
statistic_type get_stat(string run_m, string run_v){
    string data;
    ifstream fin1(run_m);
    statistic_type statistics;
    
    int size;
    getline(fin1, data);
    sscanf(data.c_str(), "# torch.Size([%d])", &size);
    statistics.size = size;
    
    statistics.running_mean.resize(size);
    for(int i=0;i<size;i++){
        getline(fin1, data);
        stringstream stream(data);
        float temp;
        stream >> temp;
        statistics.running_mean[i] = temp;
    }

    ifstream fin2(run_v);
    getline(fin2, data);
    statistics.running_variance.resize(size);
    for(int i=0;i<size;i++){
        getline(fin2, data);
        stringstream stream(data);
        float temp;
        stream >> temp;
        statistics.running_variance[i] = temp;
    }
    return statistics;
}

bn_type get_bn(string weight, string bias){
    string data;
    ifstream fin1(weight);
    bn_type bn;
    int size;

    getline(fin1, data);
    sscanf(data.c_str(), "# torch.Size([%d])", &size);
    bn.size = size;

    bn.bn_w.resize(size);
    bn.bn_b.resize(size);

    for(int i=0;i<size;i++){
        getline(fin1, data);
        stringstream stream(data);
        float temp;
        stream >> temp;
        bn.bn_w[i] = temp;
    }

    ifstream fin2(bias);
    getline(fin2, data);
    for(int i=0;i<size;i++){
        getline(fin2, data);
        stringstream stream(data);
        float temp;
        stream >> temp;
        bn.bn_b[i] = temp;
    }
    return bn;
}
#endif

void cnn::t(int s){
	for(int i=0;i<s;i++){
		for(int j=0;j<s;j++){
			cout << channel[0][i][j] << " ";
		}
        cout << "\n";
    }
}

image_type cnn::get_channel(){
    return channel;
}

fc_weight get_fc_weight(string fc_file_w){
    string data;
    fc_weight fc_w;
    int output_size, input_size;
    ifstream fin(fc_file_w);
    getline(fin, data);
    sscanf(data.c_str(), "# torch.Size([%d, %d])", &output_size, &input_size);
    fc_w.output_size = output_size;
    fc_w.input_size = input_size;

    fc_w.weight.resize(output_size);
    for(int i=0;i<output_size;i++){
        fc_w.weight[i].resize(input_size);
    }

    for(int i=0;i<output_size;i++){
        getline(fin, data);
        stringstream stream(data);
        for(int j=0;j<input_size;j++){
            float temp;
            stream >> temp;
            fc_w.weight[i][j] = temp;
        }
    }

    return fc_w;
}

fc_bias get_fc_bias(string fc_file_b){
    string data;
    ifstream fin(fc_file_b);
    fc_bias bias;
    int size;
    getline(fin, data);
    sscanf(data.c_str(), "# torch.Size([%d])", &size);
    bias.size = size;

    bias.bias.resize(size);
    for(int i=0;i<size;i++){
        getline(fin, data);
        stringstream stream(data);
        float temp;
        stream >> temp;
        bias.bias[i] = temp;
    }
    return bias;
}

fc_type fc(image_type rgb, image_type depth, fc_weight fc_w, fc_bias fc_b){
    fc_type result;
    vector<float> concat;
    result.resize(fc_w.output_size);
    concat.resize(fc_w.input_size);
    for(int i=0;i<rgb.size();i++){
        concat[i] = rgb[i][0][0];
    }
    for(int i=rgb.size();i<(rgb.size()+depth.size());i++){
        concat[i] = depth[i-64][0][0];
    }
    for(int out=0;out<fc_w.output_size;out++){
        float count = 0;
        for(int i=0;i<(rgb.size()+depth.size());i++){
            count += concat[i]*fc_w.weight[out][i];
        }
        result[out] = count+fc_b.bias[out];
    }
    return result;
}
fc_type full_connected(image_type rgb, image_type depth, string weight_file_name, string bias_file_name, int out_size){
    fc_type result;
    result.resize(out_size);
    
    fc_weight weight = get_fc_weight(weight_file_name);
    bias_type bias = get_bias(bias_file_name);
    for (int output = 0; output<result.size(); output++) {
        for (int input = 0; input < rgb.size(); input++) {
            result[output] += weight.weight[output][input]*rgb[input][0][0]+ weight.weight[output][input+rgb.size()]*depth[input][0][0];
        }
        result[output]+=bias.bias[output];
    }
    return result;
}

string ToString(int sel){
    if(sel){
        switch(sel % 2){
            case 0:
                return ".dn.body";
            case 1:
                return ".conv.0.body";
            default:
                return "";
        }
    }else{
        return ".body";
    }
}