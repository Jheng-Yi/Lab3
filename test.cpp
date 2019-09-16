#include"cnn.hpp"
#include <string.h>
#include <time.h>


int main(){
	clock_t start, end;
	start = clock();
	
	// string filename_rgb = "D:/jheng1/Gesture_recognition20190625/dataset/testing_rgb/14/P4_G015_0090.jpg";
	// string filename_depth = "D:/jheng1/Gesture_recognition20190625/dataset/testing_depth/14/P4_G015_0090.jpg";
	string dataset_root = "D:/jheng1/Gesture_recognition20190625/dataset/";
	string weight_root = "C:/Users/jheng1/Desktop/Lab3/data_32bit/";

	cnn test_RGB = cnn();
	cnn test_Depth = cnn();

	// test_RGB.r_img(filename_rgb, "RGB");
	// test_Depth.r_img(filename_depth, "Depth");

	vector<string> rgb_file = vector<string>();
    vector<string> depth_file = vector<string>();
    string image_depth = dataset_root+"testing_depth/";
    string image_rgb = dataset_root+"testing_rgb/";

	///test shuffle image
    vector<int> xxx = vector<int>();
    unsigned seed = std::chrono::system_clock::now ().time_since_epoch ().count ();
    for (int i = 0; i < 24; ++i) {
        xxx.push_back(i);
        if(i < 10){
            getdir(string(image_rgb + "0" + to_string(i) + "/"), rgb_file, seed);
            getdir(string(image_depth + "0" + to_string(i) + "/"), depth_file, seed);
        }
        else{
            getdir(string(image_rgb+ to_string(i) + "/"), rgb_file, seed);
            getdir(string(image_depth+ to_string(i) + "/"), depth_file, seed);
        }
        
    }
    std::shuffle (xxx.begin (), xxx.end (), std::default_random_engine (seed));
    std::shuffle (rgb_file.begin (), rgb_file.end (), std::default_random_engine (seed));
    std::shuffle (depth_file.begin (), depth_file.end (), std::default_random_engine (seed));

	#if (quantize)

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

	#elif (!quantize)

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
	int currect = 0;
	for(int i=0;i<24;i++){
		cout << rgb_file[i] << endl;
	}
	for(int num=0;num<24;num++){
		cout << "1\n";
		cout << rgb_file[num] << endl;
		test_RGB.r_img(rgb_file[num], "RGB");
		test_Depth.r_img(depth_file[num], "Depth");
		cout << "2\n";
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
		cout << "3\n";
		test_RGB.avgpooling();
		test_Depth.avgpooling();
		cout << "4\n";
		fc_w = get_fc_weight("./data_32bit/classifier_concat.weight");
		fc_b = get_fc_bias("./data_32bit/classifier_concat.bias");
		fc_type result;
		result = fc(test_RGB.get_channel(), test_Depth.get_channel(), get_fc_weight("./data_32bit/classifier_concat.weight"), get_fc_bias("./data_32bit/classifier_concat.bias"));
		cout << "5\n";
		int ans = 0;
		float compare = result[0];
		for(int i=0;i<24;i++){
			cout << i << " " << result[i] << endl;
			if(compare < result[i]){
				ans = i;
				compare = result[i];
			}
		}
		cout << "predict = " << ans << endl;
	}
	cout << "accuracy: " << currect/float(xxx.size()) << endl;
	cout << "Time: " << (end-start)/ CLOCKS_PER_SEC << "s" <<  endl;
	return 0;
	#endif
}