#ifndef VIDEOPROCESSOR_H
#define VIDEOPROCESSOR_H
#include <QMainWindow>
#include <QFile>
#include <QFileDialog>
#include <iostream>
#include <string>

#include "/vhd_opencv/opencv/include/opencv2/opencv.hpp"

// The frame processor interface
class FrameProcessor {
public:
// processing method
virtual void process(cv:: Mat &input, cv:: Mat &output)= 0;
};


class VideoProcessor
{
private:
    //the opencv video capture object
    cv::VideoCapture capture;

    //a bool to determine if the process callback will be called
    bool callIt;
    //input display window name
    std::string windowNameInput;
    //output display window name
    std::string windowNameOutput;
    //delay between each frame processing
    int delay;
    //number of processed frames
    long fnumber;
    //to stop the processing
    bool stop;
    //stop at this frame number
    long frameToStop;

    //
    // the OpenCV video writer object
    cv::VideoWriter writer;
    // output filename
    std::string outputFile;
    // current index for output images
    int currentIndex;
    // number of digits in output image filename
    int digits;
    // extension of output images
    std::string extension;
    // vector of image filename to be used as input
    std::vector<std::string> images;
    // image vector iterator
    std::vector<std::string>::const_iterator itImg;
   //
    // to get the next frame
    // could be: video file; camera; vector of images
    bool readNextFrame(cv::Mat&);
    // to write the output frame
    // could be: video file or images
    void writeNextFrame(cv::Mat&);
    //the callback function to be called
    //for the procession of each frame
    void (*process)(cv::Mat&, cv::Mat&);
    //class FrameProcessor: a virtual method "process" and can be
    //inherited by perticuler process classed
    FrameProcessor* frameProcessor;

public:
    // Constructor setting the default values
    VideoProcessor():callIt(false),delay(-1),fnumber(0),
        stop(false),frameToStop(-1),digits(0),
        process(0),frameProcessor(0) {}
    bool setInput(std::string);
    bool setInput(const std::vector<std::string> &);
    void displayInput(std::string);
    void displayOutput(std::string);
    void dontDisplay();
    void run();
    void stopIt();
    bool isStopped();
    bool isOpened();
    void setDelay(int);
    void callProcess();
    void dontCallProcess();
    void stopAtFrameNo(long);
    long getFrameNumber();
    long getFrameRate();
    cv::Size getFrameSize();
    int getCodec(char *);
    void setFrameProcessor(FrameProcessor* );
    void setFrameProcessor(void(*)(cv::Mat&, cv::Mat&));
    bool setOutput(const std::string &,int=0,double=0.0,bool=true);
    bool setOutput(const std::string &,const std::string &,int=3,int=0);


};

class FeatureTracker : public FrameProcessor {
        cv::Mat gray; // current gray-level image
        cv::Mat gray_prev; // previous gray-level image
        // tracked features from 0->1
        std::vector<cv::Point2f> points[2];
        // initial position of tracked points
        std::vector<cv::Point2f> initial;
        std::vector<cv::Point2f> features; // detected features
        int max_count; // maximum number of features to detect
        double qlevel; // quality level for feature detection
        double minDist; // min distance between two points
        std::vector<uchar> status; // status of tracked features
        std::vector<float> err; // error in tracking
        int middle[10];
        int objNum;
        cv::Rect objRect[10];
        public:
        FeatureTracker() : max_count(1800),
            qlevel(0.02), minDist(4.0) {}
        void process(cv::Mat &,cv::Mat &);
        void detectFeaturePoints();
        bool addNewPoints();
        bool acceptTrackedPoint(int );
        void handleTrackedPoints(cv:: Mat &);
        void drawOnImage(cv::Mat &,const std::vector<cv::Point2f> &);
        void Findobj(std::vector<cv::Point2f> & , int & );
};

class HaarCascadeTracker : public FrameProcessor {
        cv::CascadeClassifier cascade;
        std::vector<cv::Rect> detections;
        cv::Size minSize;
        cv::Size maxSize;

        public:
        HaarCascadeTracker() : minSize(cv::Size(6,6)),
            maxSize(cv::Size(64,64)) {}
        void process(cv::Mat &,cv::Mat &);
  //      bool setCascader(std::string );
 //       void handleTrackedRec(cv:: Mat &,cv:: Mat &);
};

class BGFGSegmentor : public FrameProcessor{
    cv::Mat gray;//当前帧灰度图
    cv::Mat background;//背景图，格式为32位浮点
    cv::Mat backImage;//CV_8U格式背景图
    cv::Mat foreground;//前景图
    double learningRate;//学习率
    int thresHold;//阈值，滤去扰动
    public:
    BGFGSegmentor():learningRate(0.6),thresHold(30){}
    //帧处理函数
    void process(cv::Mat &,cv::Mat &);
    void setLearnRate(long);
};


#endif // VIDEOPROCESSOR_H
