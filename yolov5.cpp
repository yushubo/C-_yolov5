#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>

using cv::Mat;
using std::cout;
using std::endl;
using std::string;
using std::vector;
// 前处理
// 推理
// 后处理:筛除置信度过低的目标，按类别分类并得到类别编号，筛除重复度过高的框，结束

// 筛除重复度过高的框策略1：拿到置信度最高的框，把剩下的去掉
// 筛除重复度过高的框策略2：拿到置信度最高的框，看剩下的框是否重复，重复过多就删除，也就是所谓的nms

// 配置参数结构体
struct Config {
    float conf_threshold = 0.7;    // 置信度阈值
    float nms_threshold = 0.4;     // NMS阈值
    int input_width = 640;         // 输入图像宽度
    int input_height = 640;        // 输入图像高度
    string model_path = "../best_lv.onnx";  // 模型路径
    string image_path = "../1383.jpg";      // 测试图像路径
};

// 类别名称
static const vector<string> class_name = {"zhen_kong", "ca_shang", "zang_wu", "zhe_zhou"};

// 错误处理函数
void checkError(bool condition, const string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

// 打印检测结果  , 用于调试查看yolo输出
void print_result(const Mat &result, float conf = 0.7, int len_data = 9)
{
    float *pdata = (float *)result.data;
    for (int i = 0; i < result.total() / len_data; i++)
    {
        if (pdata[4] > conf)
        {
            for (int j = 0; j < len_data; j++)
            {
                cout << pdata[j] << " ";
            }
            cout << endl;
        }
        pdata += len_data;
    }
    return;
}

// 筛选出置信度高于指定阈值（conf）的检测结果
vector<vector<float>> get_info(const Mat &result, float conf = 0.7, int len_data = 9)
{
    float *pdata = (float *)result.data;
    vector<vector<float>> info;
    for (int i = 0; i < result.total() / len_data; i++)
    {
        if (pdata[4] > conf)
        {
            vector<float> info_line;
            for (int j = 0; j < len_data; j++)
            {
                // cout << pdata[j] << " ";
                info_line.push_back(pdata[j]);
            }
            // cout << endl;
            info.push_back(info_line);
        }
        pdata += len_data;
    }
    return info;
}

// 坐标转换，变成左上角和右下角坐标
void info_simplify(vector<vector<float>> &info)
{
    for (auto i = 0; i < info.size(); i++)
    {
        // 从第 5 个元素开始 找最大值，让它成为第六个数
        info[i][5] = std::max_element(info[i].cbegin() + 5, info[i].cend()) - (info[i].cbegin() + 5);
        info[i].resize(6);  //调整数组长度
        float x = info[i][0];
        float y = info[i][1];
        float w = info[i][2];
        float h = info[i][3];
        info[i][0] = x - w / 2.0;
        info[i][1] = y - h / 2.0;
        info[i][2] = x + w / 2.0;
        info[i][3] = y + h / 2.0;
    }

    // 调试查看输出，处理结果
    cout << "Simplified Info:" << endl;
    for (const auto &row : info)
    {
        for (const auto &val : row)
        {
            cout << val << " ";
        }
        cout << endl;
    }
}

// 创建一个三维数组，每个二维数组存储了同一类别的检测结果。 用于将类别分组存放
vector<vector<vector<float>>> split_info(vector<vector<float>> &info)
{
    vector<vector<vector<float>>> info_split;    // 用于存储分组后的检测结果 三维数组
    vector<int> class_id;                       // 用于存储已经处理过的类别 ID
    for (auto i = 0; i < info.size(); i++)
    {
        if (std::find(class_id.begin(), class_id.end(), (int)info[i][5]) == class_id.end())
        {
            class_id.push_back((int)info[i][5]);
            vector<vector<float>> info_;
            info_split.push_back(info_);
        }
        info_split[std::find(class_id.begin(), class_id.end(), (int)info[i][5]) - class_id.begin()].push_back(info[i]);
    }
    return info_split;
}

// 优化后的NMS实现
void nms(vector<vector<float>>& info, float iou_threshold = 0.4) {
    if (info.empty()) return;
    
    // 按置信度排序
    std::sort(info.begin(), info.end(), 
        [](const vector<float>& a, const vector<float>& b) { return a[4] > b[4]; });
    
    vector<vector<float>> result;
    vector<bool> suppressed(info.size(), false);
    
    for (size_t i = 0; i < info.size(); ++i) {
        if (suppressed[i]) continue;
        
        result.push_back(info[i]);
        
        for (size_t j = i + 1; j < info.size(); ++j) {
            if (suppressed[j]) continue;
            
            // 计算IoU
            float x1 = std::max(info[i][0], info[j][0]);
            float y1 = std::max(info[i][1], info[j][1]);
            float x2 = std::min(info[i][2], info[j][2]);
            float y2 = std::min(info[i][3], info[j][3]);
            
            float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
            float area1 = (info[i][2] - info[i][0]) * (info[i][3] - info[i][1]);
            float area2 = (info[j][2] - info[j][0]) * (info[j][3] - info[j][1]);
            float iou = intersection / (area1 + area2 - intersection);
            
            if (iou > iou_threshold) {
                suppressed[j] = true;
            }
        }
    }
    
    info = result;
}

void print_info(const vector<vector<float>> &info)
{
    for (auto i = 0; i < info.size(); i++)
    {
        for (auto j = 0; j < info[i].size(); j++)
        {
            cout << info[i][j] << " ";
        }
        cout << endl;
    }
}

void draw_box(Mat &img, const vector<vector<float>> &info)
{
    for (int i = 0; i < info.size(); i++)
    {
        cv::rectangle(img, cv::Point(info[i][0], info[i][1]), cv::Point(info[i][2], info[i][3]), cv::Scalar(0, 255, 0));
        string label;
        label += class_name[info[i][5]];
        label += "  ";
        std::stringstream oss;
        oss << info[i][4];
        label += oss.str();
        cv::putText(img, label, cv::Point(info[i][0], info[i][1]), 1, 2, cv::Scalar(0, 255, 0), 2);

    }
}

int main() {
    try {
        Config config;
        
        // 加载模型
        cv::dnn::Net net = cv::dnn::readNetFromONNX(config.model_path);
        checkError(!net.empty(), "Failed to load model");
        
        // 读取图像
        Mat img = cv::imread(config.image_path);
        checkError(!img.empty(), "Failed to load image");
        
        // 图像预处理
        cv::resize(img, img, cv::Size(config.input_width, config.input_height));
        Mat blob = cv::dnn::blobFromImage(img, 1.0/255.0, cv::Size(config.input_width, config.input_height), 
                                        cv::Scalar(), true, false);
        
        // 模型推理
        net.setInput(blob);
        vector<string> out_name = {"output0"};
        vector<Mat> netoutput;
        net.forward(netoutput, out_name);
        
        // 后处理
        Mat result = netoutput[0];
        vector<vector<float>> info = get_info(result, config.conf_threshold);
        info_simplify(info);
        vector<vector<vector<float>>> info_split = split_info(info);
        
        // 对每个类别进行NMS和绘制
        for (size_t i = 0; i < info_split.size(); i++) {
            nms(info_split[i], config.nms_threshold);
            draw_box(img, info_split[i]);
        }
        
        // 显示结果
        cv::imshow("Detection Result", img);
        cv::waitKey(0);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
