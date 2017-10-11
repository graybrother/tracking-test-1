#ifndef HARRISDETECTOR_H
#define HARRISDETECTOR_H

class HarrisDetector
{
private:
    // 32-bit float image of corner strength
    cv::Mat cornerStrength;
    // 32-bit float image of thresholded corners
    cv::Mat cornerTh;
    // image of local maxima (internal)
    cv::Mat localMax;
    // size of neighborhood for derivatives smoothing
    int neighbourhood;
    // aperture for gradient computation
    int aperture;
    // Harris parameter
    double k;
    // maximum strength for threshold computation
    double maxStrength;
    // calculated threshold (internal)
    double threshold;
    // size of neighborhood for non-max suppression
    int nonMaxSize;
    // kernel for non-max suppression
    cv::Mat kernel;

public:
    HarrisDetector():neighbourhood(3), aperture(3),
    k(0.01), maxStrength(0.0),
    threshold(0.01), nonMaxSize(3) {}
    // create kernel used in non-maxima suppression
    void setLocalMaxWindowSize(int);
    void detect(const cv::Mat &);
    cv::Mat getCornerMap(double);
    void getCorners(std::vector<cv::Point> &, double);
    void getCorners(std::vector<cv::Point> &, const cv::Mat &);
    //
    void drawOnImage(cv::Mat&, const std::vector<cv::Point>&);

};

#endif // HARRISDETECTOR_H
