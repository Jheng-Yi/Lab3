#include"cnn.hpp"
#include <string.h>
#include <time.h>


int main(){
	clock_t start, end;
	start = clock();
	
	string filename_rgb = "D:/jheng1/Gesture_recognition20190625/dataset/testing_rgb/00/P3_G00_0258.jpg";
	string filename_depth = "D:/jheng1/Gesture_recognition20190625/dataset/testing_depth/00/P3_G00_0258.jpg";
	
	cnn test_RGB = cnn();
	cnn test_Depth = cnn();

	test_RGB.r_img(filename_rgb, "RGB");
	test_Depth.r_img(filename_depth, "Depth");

	#if (quantize == 1)

	kernel_type kernel;
	bias_type bias;

	string weight_para_path;
	string bias_para_path;

	fc_weight fc_w;
	fc_bias fc_b;

	for(int cnt=0;cnt<9;cnt++){
		//RGB
		weight_para_path = "./data_8bit/rgb_feature."+to_string((cnt+1)/2)+ToString(cnt)+".conv.weight";
		bias_para_path = "./data_8bit/rgb_feature."+to_string((cnt+1)/2)+ToString(cnt)+".conv.bias";

		kernel = get_kernel(weight_para_path);
		bias = get_bias(bias_para_path);
		cout << cnt+1 << " conv" << endl;
		test_RGB.conv(kernel, bias, 1);
		if(cnt % 2 == 0){
			test_RGB.maxpooling(2, 2);
		}

		//Depth
		weight_para_path = "./data_8bit/depth_feature."+to_string((cnt+1)/2)+ToString(cnt)+".conv.weight";
		bias_para_path = "./data_8bit/depth_feature."+to_string((cnt+1)/2)+ToString(cnt)+".conv.bias";

		kernel = get_kernel(weight_para_path);
		bias = get_bias(bias_para_path);
		test_Depth.conv(kernel, bias, 1);
		if(cnt % 2 == 0){
			test_Depth.maxpooling(2, 2);
		}
	}

	test_RGB.avgpooling();
	test_Depth.avgpooling();

	fc_w = get_fc_weight("./data_8bit/classifier_concat.weight");
	fc_b = get_fc_bias("./data_8bit/classifier_concat.bias");
	fc_type result;
	result = fc(test_RGB.get_channel(), test_Depth.get_channel(), fc_w, fc_b);

	int ans = 0;
	float compare = result[0];
	for(int i=0;i<24;i++){
		cout << i << " " << result[i] << endl;
		if(compare < result[i]){
			ans = i;
			compare = result[i];
		}
	}
	cout << "ans = " << ans << endl;

	return 0;

	#elif (quantize == 0)

	kernel_type kernel;
	bias_type running_mean;
	bias_type running_var;
	bias_type bn_weight;
	bias_type bn_bias;

	//RGB
	string para_rgb;
	string run_m_rgb;
	string run_v_rgb;
	//Depth
	string para_depth;
	string run_m_depth;
	string run_v_depth;
	//batch normalization
	string bn_w; //bn_weight
	string bn_b; //bn_bias

	fc_weight fc_w;
	fc_bias fc_b;
	#if (debug)
	int i = 1;
	int j = 1;
	#endif

	for(int cnt=0;cnt<9;cnt++){
		//RGB
		para_rgb = "./data_32bit/rgb_feature."+to_string((cnt+1)/2)+ToString(cnt)+".conv.weight";
		run_m_rgb = "./data_32bit/rgb_feature."+to_string((cnt+1)/2)+ToString(cnt)+".bn.running_mean";
		run_v_rgb = "./data_32bit/rgb_feature."+to_string((cnt+1)/2)+ToString(cnt)+".bn.running_var";
		bn_w = "./data_32bit/rgb_feature."+to_string((cnt+1)/2)+ToString(cnt)+".bn.weight";
		bn_b = "./data_32bit/rgb_feature."+to_string((cnt+1)/2)+ToString(cnt)+".bn.bias";

		kernel = get_kernel(para_rgb);
		running_mean = get_bias(run_m_rgb);
		running_var = get_bias(run_v_rgb);
		bn_weight = get_bias(bn_w);
		bn_bias = get_bias(bn_b);
		cout << cnt+1 << " conv" << endl;
		test_RGB.conv(kernel, running_mean, running_var, bn_weight, bn_bias, 1);
		if(cnt % 2 == 0){
			test_RGB.maxpooling(2, 2);
		}

		//Depth
		para_depth = "./data_32bit/depth_feature."+to_string((cnt+1)/2)+ToString(cnt)+".conv.weight";
		run_m_depth = "./data_32bit/depth_feature."+to_string((cnt+1)/2)+ToString(cnt)+".bn.running_mean";
		run_v_depth = "./data_32bit/depth_feature."+to_string((cnt+1)/2)+ToString(cnt)+".bn.running_var";
		bn_w = "./data_32bit/depth_feature."+to_string((cnt+1)/2)+ToString(cnt)+".bn.weight";
		bn_b = "./data_32bit/depth_feature."+to_string((cnt+1)/2)+ToString(cnt)+".bn.bias";

		kernel = get_kernel(para_depth);
		running_mean = get_bias(run_m_depth);
		running_var = get_bias(run_v_depth);
		bn_weight = get_bias(bn_w);
		bn_bias = get_bias(bn_b);
		test_Depth.conv(kernel, running_mean, running_var, bn_weight, bn_bias, 1);
		if(cnt % 2 == 0){
			test_Depth.maxpooling(2, 2);
		}
	}

	test_RGB.avgpooling();
	test_Depth.avgpooling();

	fc_w = get_fc_weight("./data_32bit/classifier_concat.weight");
	fc_b = get_fc_bias("./data_32bit/classifier_concat.bias");
	fc_type result;
	result = fc(test_RGB.get_channel(), test_Depth.get_channel(), get_fc_weight("./data_32bit/classifier_concat.weight"), get_fc_bias("./data_32bit/classifier_concat.bias"));

	int ans = 0;
	float compare = result[0];
	for(int i=0;i<24;i++){
		cout << i << " " << result[i] << endl;
		if(compare < result[i]){
			ans = i;
			compare = result[i];
		}
	}
	cout << "ans = " << ans << endl;

	return 0;

	#endif
}