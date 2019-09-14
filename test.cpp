#include"cnn.hpp"
#include <string.h>
#include <time.h>


int main(){
	clock_t start, end;
	start = clock();
	#if (quantize == 1)
	//8-bit quantize parameter
	char para_rgb[] = "rgb_feature.0.body.conv.weight rgb_feature.0.body.conv.bias rgb_feature.1.conv.0.body.conv.weight rgb_feature.1.conv.0.body.conv.bias rgb_feature.1.dn.body.conv.weight rgb_feature.1.dn.body.conv.bias rgb_feature.2.conv.0.body.conv.weight rgb_feature.2.conv.0.body.conv.bias rgb_feature.2.dn.body.conv.weight rgb_feature.2.dn.body.conv.bias rgb_feature.3.conv.0.body.conv.weight rgb_feature.3.conv.0.body.conv.bias rgb_feature.3.dn.body.conv.weight rgb_feature.3.dn.body.conv.bias rgb_feature.4.conv.0.body.conv.weight rgb_feature.4.conv.0.body.conv.bias rgb_feature.4.dn.body.conv.weight rgb_feature.4.dn.body.conv.bias";
    char para_depth[] = "depth_feature.0.body.conv.weight depth_feature.0.body.conv.bias depth_feature.1.conv.0.body.conv.weight depth_feature.1.conv.0.body.conv.bias depth_feature.1.dn.body.conv.weight depth_feature.1.dn.body.conv.bias depth_feature.2.conv.0.body.conv.weight depth_feature.2.conv.0.body.conv.bias depth_feature.2.dn.body.conv.weight depth_feature.2.dn.body.conv.bias depth_feature.3.conv.0.body.conv.weight depth_feature.3.conv.0.body.conv.bias depth_feature.3.dn.body.conv.weight depth_feature.3.dn.body.conv.bias depth_feature.4.conv.0.body.conv.weight depth_feature.4.conv.0.body.conv.bias depth_feature.4.dn.body.conv.weight depth_feature.4.dn.body.conv.bias";
	//32-bit parameter
	//char para_rgb[] = "rgb_feature.0.body.conv.weight rgb_feature.0.body.bn.weight rgb_feature.0.body.bn.bias rgb_feature.0.body.bn.running_mean rgb_feature.0.body.bn.running_var rgb_feature.1.conv.0.body.conv.weight rgb_feature.1.conv.0.body.bn.weight rgb_feature.1.conv.0.body.bn.bias rgb_feature.1.conv.0.body.bn.running_mean rgb_feature.1.conv.0.body.bn.running_var rgb_feature.1.dn.body.conv.weight rgb_feature.1.dn.body.bn.weight rgb_feature.1.dn.body.bn.bias rgb_feature.1.dn.body.bn.running_mean rgb_feature.1.dn.body.bn.running_var rgb_feature.2.conv.0.body.conv.weight rgb_feature.2.conv.0.body.bn.weight rgb_feature.2.conv.0.body.bn.bias rgb_feature.2.conv.0.body.bn.running_mean rgb_feature.2.conv.0.body.bn.running_var rgb_feature.2.dn.body.conv.weight rgb_feature.2.dn.body.bn.weight rgb_feature.2.dn.body.bn.bias rgb_feature.2.dn.body.bn.running_mean rgb_feature.2.dn.body.bn.running_var rgb_feature.3.conv.0.body.conv.weight rgb_feature.3.conv.0.body.bn.weight rgb_feature.3.conv.0.body.bn.bias rgb_feature.3.conv.0.body.bn.running_mean rgb_feature.3.conv.0.body.bn.running_var rgb_feature.3.dn.body.conv.weight rgb_feature.3.dn.body.bn.weight rgb_feature.3.dn.body.bn.bias rgb_feature.3.dn.body.bn.running_mean rgb_feature.3.dn.body.bn.running_var rgb_feature.4.conv.0.body.conv.weight rgb_feature.4.conv.0.body.bn.weight rgb_feature.4.conv.0.body.bn.bias rgb_feature.4.conv.0.body.bn.running_mean rgb_feature.4.conv.0.body.bn.running_var rgb_feature.4.dn.body.conv.weight rgb_feature.4.dn.body.bn.weight rgb_feature.4.dn.body.bn.bias rgb_feature.4.dn.body.bn.running_mean rgb_feature.4.dn.body.bn.running_var";	
	//char para_depth[] = "depth_feature.0.body.conv.weight depth_feature.0.body.bn.weight depth_feature.0.body.bn.bias depth_feature.0.body.bn.running_mean depth_feature.0.body.bn.running_var depth_feature.1.conv.0.body.conv.weight depth_feature.1.conv.0.body.bn.weight depth_feature.1.conv.0.body.bn.bias depth_feature.1.conv.0.body.bn.running_mean depth_feature.1.conv.0.body.bn.running_var depth_feature.1.dn.body.conv.weight depth_feature.1.dn.body.bn.weight depth_feature.1.dn.body.bn.bias depth_feature.1.dn.body.bn.running_mean depth_feature.1.dn.body.bn.running_var depth_feature.2.conv.0.body.conv.weight depth_feature.2.conv.0.body.bn.weight depth_feature.2.conv.0.body.bn.bias depth_feature.2.conv.0.body.bn.running_mean depth_feature.2.conv.0.body.bn.running_var depth_feature.2.dn.body.conv.weight depth_feature.2.dn.body.bn.weight depth_feature.2.dn.body.bn.bias depth_feature.2.dn.body.bn.running_mean depth_feature.2.dn.body.bn.running_var depth_feature.3.conv.0.body.conv.weight depth_feature.3.conv.0.body.bn.weight depth_feature.3.conv.0.body.bn.bias depth_feature.3.conv.0.body.bn.running_mean depth_feature.3.conv.0.body.bn.running_var depth_feature.3.dn.body.conv.weight depth_feature.3.dn.body.bn.weight depth_feature.3.dn.body.bn.bias depth_feature.3.dn.body.bn.running_mean depth_feature.3.dn.body.bn.running_var depth_feature.4.conv.0.body.conv.weight depth_feature.4.conv.0.body.bn.weight depth_feature.4.conv.0.body.bn.bias depth_feature.4.conv.0.body.bn.running_mean epth_feature.4.conv.0.body.bn.running_var depth_feature.4.dn.body.conv.weight depth_feature.4.dn.body.bn.weight depth_feature.4.dn.body.bn.bias depth_feature.4.dn.body.bn.running_mean depth_feature.4.dn.body.bn.running_var";
	char* p_rgb = NULL;
	char* p_depth = NULL;
    char* del = " ";
	char pwd_rgb[100] = "./data_8bit/";
	char pwd_depth[100] = "./data_8bit/";
	char* p_save_rgb = NULL;
	char* p_save_depth = NULL;
	string filename_rgb = "D:/jheng1/Gesture_recognition20190625/dataset/testing_rgb/10/P1_G011_0176.jpg";
	string filename_depth = "D:/jheng1/Gesture_recognition20190625/dataset/testing_depth/10/P1_G011_0176.jpg";
	cnn test_RGB = cnn();
	cnn test_Depth = cnn();
	test_RGB.r_img(filename_rgb, "RGB");
	test_Depth.r_img(filename_depth, "Depth");
	kernel_type RGB_kernel;
	kernel_type Depth_kernel;
	bias_type RGB_bias;
	bias_type Depth_bias;
	
	// RGB
	p_rgb = strtok_r(para_rgb, del, &p_save_rgb);
	strcat(pwd_rgb, p_rgb);
    RGB_kernel = get_kernel(pwd_rgb);
	//cout << pwd_rgb << endl;
	pwd_rgb[8] = '\0';
	p_rgb = strtok_r(NULL, del, &p_save_rgb);
	strcat(pwd_rgb, p_rgb);
	RGB_bias = get_bias(pwd_rgb);
	//cout << pwd_rgb << endl;
	pwd_rgb[8] = '\0';
	test_RGB.conv(RGB_kernel, RGB_bias, 1);
	test_RGB.maxpooling(2, 2);
	for(int i=0;i<4;i++){
	// 1st conv
		p_rgb = strtok_r(NULL, del, &p_save_rgb);
		strcat(pwd_rgb, p_rgb);
		RGB_kernel = get_kernel(pwd_rgb);
		//cout << pwd_rgb << endl;
		pwd_rgb[8] = '\0';
		p_rgb = strtok_r(NULL, del, &p_save_rgb);
		strcat(pwd_rgb, p_rgb);
		RGB_bias = get_bias(pwd_rgb);
		//cout << pwd_rgb << endl;
		pwd_rgb[8] = '\0';
		test_RGB.conv(RGB_kernel, RGB_bias, 1);
	// 2nd conv
		p_rgb = strtok_r(NULL, del, &p_save_rgb);
		strcat(pwd_rgb, p_rgb);
		RGB_kernel = get_kernel(pwd_rgb);
		//cout << pwd_rgb << endl;
		pwd_rgb[8] = '\0';
		p_rgb = strtok_r(NULL, del, &p_save_rgb);
		strcat(pwd_rgb, p_rgb);
		RGB_bias = get_bias(pwd_rgb);
		//cout << pwd_rgb << endl;
		pwd_rgb[8] = '\0';
		test_RGB.conv(RGB_kernel, RGB_bias, 1);
		test_RGB.maxpooling(2, 2);
	}
	test_RGB.avgpooling();

	// Depth
	p_depth = strtok_r(para_depth, del, &p_save_depth);
	strcat(pwd_depth, p_depth);
	Depth_kernel = get_kernel(pwd_depth);
	//cout << pwd_depth<< endl;
	pwd_depth[8] = '\0';
	p_depth = strtok_r(NULL, del, &p_save_depth);
	strcat(pwd_depth, p_depth);
	Depth_bias = get_bias(pwd_depth);
	//cout << pwd_depth << endl;
	pwd_depth[8] = '\0';
	test_Depth.conv(Depth_kernel, Depth_bias, 1);
	test_Depth.maxpooling(2, 2);
	for(int i=0;i<4;i++){
	//1st conv
		p_depth = strtok_r(NULL, del, &p_save_depth);
		strcat(pwd_depth, p_depth);
		Depth_kernel = get_kernel(pwd_depth);
		//cout << pwd_depth << endl;
		pwd_depth[8] = '\0';
		p_depth = strtok_r(NULL, del, &p_save_depth);
		strcat(pwd_depth, p_depth);
		Depth_bias = get_bias(pwd_depth);
		//cout << pwd_depth << endl;
		pwd_depth[8] = '\0';
		test_Depth.conv(Depth_kernel, Depth_bias, 1);
	//2nd conv
		p_depth = strtok_r(NULL, del, &p_save_depth);
		strcat(pwd_depth, p_depth);
		Depth_kernel = get_kernel(pwd_depth);
		//cout << pwd_depth << endl;
		pwd_depth[8] = '\0';
		p_depth = strtok_r(NULL, del, &p_save_depth);
		strcat(pwd_depth, p_depth);
		Depth_bias = get_bias(pwd_depth);
		//cout << pwd_depth << endl;
		pwd_depth[8] = '\0';
		test_Depth.conv(Depth_kernel, Depth_bias, 1);
		test_Depth.maxpooling(2, 2);
	}
	test_Depth.avgpooling();
	
	//FC_Layer
	fc_weight fc_w;
	fc_bias fc_b;
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
	end = clock();
	double diff = end - start;
	cout << "ans = " << ans << endl;
	cout << "run time = " << diff  << "ms" << endl;

	return 0;

	#elif (quantize == 0)
	string filename_rgb = "D:/jheng1/Gesture_recognition20190625/dataset/testing_rgb/08/P1_G08_0004.jpg";
	string filename_depth = "D:/jheng1/Gesture_recognition20190625/dataset/testing_depth/08/P1_G08_0004.jpg";
	cnn test_RGB = cnn();
	cnn test_Depth = cnn();

	test_RGB.r_img(filename_rgb, "RGB");
	test_Depth.r_img(filename_depth, "Depth");

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

		if(cnt == 8){
			for(int i=0;i<32;i++){
				cout << running_mean.bias[i] << endl;
			}
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
		cout << cnt+1 << "conv" << endl;
		test_Depth.conv(kernel, running_mean, running_var, bn_weight, bn_bias, 1);
		if(cnt % 2 == 0){
			test_Depth.maxpooling(2, 2);
		}

		if(cnt == 8){
			for(int i=0;i<32;i++){
				cout << running_mean.bias[i] << endl;
			}
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