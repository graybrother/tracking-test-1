#include "videoprocessor.h"
#include "mainwindow.h"
#include "ui_mainwindow.h"



// processing method
void BGFGSegmentor::process(cv:: Mat &frame, cv:: Mat &output) {
        // convert to gray-level image
        cv::cvtColor(frame, gray, CV_BGR2GRAY);
        // initialize background to 1st frame
        if (background.empty())
                gray.convertTo(background, CV_32F);
        // convert background to 8U
        background.convertTo(backImage,CV_8U);
        // compute difference between image and background
        cv::absdiff(backImage,gray,foreground);
        // apply threshold to foreground image
        cv::threshold(foreground,output,
        thresHold,255,cv::THRESH_BINARY_INV);
        // accumulate background
        cv::accumulateWeighted(gray, background,
        learningRate, output);
}

void BGFGSegmentor::setLearnRate(long learnRate){
    learningRate=learnRate;
    return;
}

