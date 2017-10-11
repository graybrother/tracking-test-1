#ifndef HOGSVM_H
#define HOGSVM_H

void drawHOG(std::vector<float>::const_iterator,int,cv::Mat &,float);
void drawHOGDescriptors(const cv::Mat &,cv::Mat &,cv::Size,int);

#endif // HOGSVM_H
