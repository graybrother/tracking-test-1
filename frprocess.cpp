#include "mainwindow.h"
#include "ui_mainwindow.h"
//#include "process.h"

void canny(cv::Mat& img, cv::Mat& out) {
    using namespace cv;
    //Comvert to gray
    if(img.channels()==3)
        cvtColor(img,out,CV_BGR2GRAY);
    //compute Canny edges
    Canny(out,out,100,200);
    //Invert the image
}





/*
int otsu (cv::Mat &img)
{
   int _low,_top,mbest=0;
   float mn = img->height*img->width;
   InitPixel(img,_low,_top);
   float max_otsu = 0;
   mbest = 0;
   if( _low == _top)
       mbest = _low;
   else
   {
       for(int i = _low; i< _top ; i++)
       {
           float w0 = (float)((Num[_top]-Num[i]) / mn);
           float w1 = 1 - w0;
           float u0 = (float)((Sum[_top]-Sum[i])/(Num[_top]-Num[i]));
           float u1 = (float)(Sum[i]/Num[i]);
           float u = w0*u0 + w1*u1;
           float g = w0*(u0 - u)*(u0 - u) + w1*(u1 - u)*(u1 - u);
           if( g > max_otsu)
           {
               mbest = i;
               max_otsu = g;
           }
       }
   }
   return mbest;
*/
