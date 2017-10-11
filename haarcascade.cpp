#include "videoprocessor.h"
#include "mainwindow.h"
#include "ui_mainwindow.h"

void HaarCascadeTracker::process(Mat &inputImage, Mat &outputImage){
    if (!cascade.load("haarcascades/haarcascade_frontalface_alt.xml")) {
        std::cout << "Error when loading the face cascade classfier!" << std::endl;
        return ;
    }

    cascade.detectMultiScale(inputImage, // input image
        detections, // detection results
        1.1,        // scale reduction factor
        3,          // number of required neighbor detections
        0,          // flags (not used)
        minSize,    // minimum object size to be detected
        maxSize); // maximum object size to be detected

    inputImage.copyTo(outputImage);
    // draw detections on image
    for (int i = 0; i < detections.size(); i++)
        cv::rectangle(outputImage, detections[i], cv::Scalar(255, 255, 255), 2);

}
