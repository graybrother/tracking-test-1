#ifndef OBJFEATURE_H
#define OBJFEATURE_H

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "opencv.hpp"

#define TrustThres  3
#define MAXOBJNUM   4
#define MOVENUM     10

typedef struct object_Feature
{
    int objID;
    int trustedValue;
    int notmoveCount;
    bool isCurrentObj;

   //for correspondent Rect from vibe
    int rectIndex;
    //points for mediantrack
    std::vector<cv::Point2f> points[2];

    //data of tracking result
    int trackedPointNum;
    cv::Rect rect;
    cv::Point2f center;
    int mvIndex;
    double objMvX[MOVENUM];
    double objMvY[MOVENUM];
    double moveX;    //filted objMvX
    double moveY;    //filted objMvY
}object_Feature_t;

void object_Feature_Init(object_Feature_t &);
bool isMatchedRect(cv::Rect &,cv::Rect &);
bool isSameRect(cv::Rect &,cv::Rect &);
cv::Rect getRect(std::vector<cv::Point2f> &);
void BubbleSort(std::vector<cv::Rect> &, int );

#endif // OBJFEATURE_H
