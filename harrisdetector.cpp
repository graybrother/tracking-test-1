#include "mainwindow.h"
#include "harrisdetector.h"

//HarrisDetector::HarrisDetector()
//{
//}
// Compute Harris corners
void HarrisDetector::detect(const cv::Mat& image) {
        // Harris computation
        cv::cornerHarris(image,cornerStrength,
        neighbourhood,// neighborhood size
        aperture, // aperture size
        k); // Harris parameter
        // internal threshold computation
        double minStrength; // not used
        cv::minMaxLoc(cornerStrength,
        &minStrength,&maxStrength);
        // local maxima detection
        cv::Mat dilated; // temporary image
        cv::dilate(cornerStrength,dilated,cv::Mat());
        cv::compare(cornerStrength,dilated,
        localMax,cv::CMP_EQ);
}

// Get the corner map from the computed Harris values
cv::Mat HarrisDetector::getCornerMap(double qualityLevel) {
        cv::Mat cornerMap;
        // thresholding the corner strength
        threshold= qualityLevel*maxStrength;

        std::cout << maxStrength << " threshold used to get cornerMap\n";

        cv::threshold(cornerStrength,cornerTh,
        threshold,255,cv::THRESH_BINARY);

        // convert to 8-bit image
        cornerTh.convertTo(cornerMap,CV_8U);
        // non-maxima suppression
        cv::bitwise_and(cornerMap,localMax,cornerMap);
        return cornerMap;
}

// Get the feature points from the computed Harris values
void HarrisDetector::getCorners(std::vector<cv::Point> &points,
        double qualityLevel) {
        // Get the corner map
        cv::Mat cornerMap= getCornerMap(qualityLevel);
        // Get the corners
        getCorners(points, cornerMap);
}
// Get the feature points from the computed corner map
void HarrisDetector::getCorners(std::vector<cv::Point> &points,
        const cv::Mat& cornerMap) {
        // Iterate over the pixels to obtain all features
        for( int y = 0; y < cornerMap.rows; y++ ) {
        const uchar* rowPtr = cornerMap.ptr<uchar>(y);
        for( int x = 0; x < cornerMap.cols; x++ ) {
            // if it is a feature point
            if (rowPtr[x]) {
                points.push_back(cv::Point(x,y));
                }
            }
        }
}

// Draw circles at feature point locations on an image
void HarrisDetector::drawOnImage(cv::Mat &image,
                const std::vector<cv::Point> &points){
        cv::Scalar color= cv::Scalar(255,255,255);
        int radius=3;
        int thickness=1;
        std::vector<cv::Point>::const_iterator it=
        points.begin();
        // for all corners
        while (it!=points.end()) {
        // draw a circle at each corner location
        cv::circle(image,*it,radius,color,thickness);
        ++it;
        }
}

void HarrisDetector::setLocalMaxWindowSize(int nonMaxSize){
        this->nonMaxSize = nonMaxSize;
}

