#include "videoprocessor.h"
#include "mainwindow.h"
#include "ui_mainwindow.h"


void FeatureTracker::process(cv:: Mat &frame, cv:: Mat &output) {
        // convert to gray-level image
        cv::cvtColor(frame, gray, CV_RGB2GRAY);
        frame.copyTo(output);

        //高斯滤波
          //      GaussianBlur( gray, gray, Size( 5, 5 ), 0, 0 );

        //直方图均衡化
     //          equalizeHist( gray, gray );
        // 1. if new feature points must be added
        if(addNewPoints())
        {
        // detect feature points
        detectFeaturePoints();
        // add the detected features to
        // the currently tracked features
        points[0].insert(points[0].end(),
        features.begin(),features.end());
        initial.insert(initial.end(),
        features.begin(),features.end());
        }
        // for first image of the sequence
        if(gray_prev.empty())
        gray.copyTo(gray_prev);
        // 2. track features
        cv::calcOpticalFlowPyrLK(
             gray_prev, gray, // 2 consecutive images
              points[0], // input point positions in first image
              points[1], // output point positions in the 2nd image
               status, // tracking success
               err); // tracking error
        // 2. loop over the tracked points to reject some
        int k=0;
        for( int i= 0; i < points[1].size(); i++ ) {
        // do we keep this point?
        if (acceptTrackedPoint(i)) {
        // keep this point in vector
        initial[k]= initial[i];
        points[1][k++] = points[1][i];
         }
        }
        // eliminate unsuccesful points
        points[1].resize(k);
        initial.resize(k);
        // 3. handle the accepted tracked points
        handleTrackedPoints(output);
        if(points[1].size()>10)
            Findobj(points[1],objNum);
        for(k=0; k<objNum; k++)
             cv::rectangle(output, objRect[k], cv::Scalar(255, 255, 255), 2);
        // 4. current points and image become previous ones
        std::swap(points[1], points[0]);
        cv::swap(gray_prev, gray);
}

/*
void FeatureTracker::process(cv:: Mat &frame, cv:: Mat &output) {
        // convert to gray-level image
        Mat roi;
        cv::cvtColor(frame, roi, CV_RGB2GRAY);
        frame.copyTo(output);

        gray=roi(Rect(20,60,600,190));
 //高斯滤波
        GaussianBlur( gray, gray, Size( 5, 5 ), 0, 0 );

//直方图均衡化
       equalizeHist( gray, gray );
        // detect feature points
        detectFeaturePoints();

        // draw featurepoint
        drawOnImage(output,features);

}
*/
// feature point detection
void FeatureTracker::detectFeaturePoints() {
        // detect the features
        cv::goodFeaturesToTrack(gray, // the image
        features, // the output detected features
        max_count, // the maximum number of features
        qlevel, // quality level
        minDist); // min distance between two features
}

// determine if new points should be added
bool FeatureTracker::addNewPoints() {
        // if too few points
        return points[0].size()<=25;
}

// determine which tracked point should be accepted
bool FeatureTracker::acceptTrackedPoint(int i) {
return status[i] &&
// if point has moved
(abs(points[0][i].x-points[1][i].x)+
(abs(points[0][i].y-points[1][i].y))>1);
}

// handle the currently tracked points
void FeatureTracker::handleTrackedPoints(cv:: Mat &output) {
        // for all tracked points
        for(int i= 0; i < points[1].size(); i++ ) {
        // draw line and circle
        cv::line(output,
        initial[i], // initial position
        points[1][i],// new position
        cv::Scalar(255,255,255));
        cv::circle(output, points[1][i], 3,
        cv::Scalar(255,255,255),-1);
        }
}
// Draw circles at feature point locations on an image
void FeatureTracker::drawOnImage(cv::Mat &image,
                const std::vector<cv::Point2f> &points){
        cv::Scalar color= cv::Scalar(255,255,255);
        int radius=3;
        int thickness=1;
        std::vector<cv::Point2f>::const_iterator it=
        points.begin();
        // for all corners
        while (it!=points.end()) {
        // draw a circle at each corner location
        cv::circle(image,*it,radius,color,thickness);
        ++it;
        }
        std::cout<<"  feature points  "<<features.size();
}
void FeatureTracker::Findobj(std::vector<cv::Point2f> &point , int &objNum)
{
    int T = 10;
    int Num[10];
    int Sum[10];
    int round_1_Num=0;
    int round_2_Num=0;

    memset(Num,0,sizeof(Num));
    memset(Sum,0,sizeof(Sum));
  //  memset(middle,0,sizeof(middle));
    objNum=0;
   int recleft[10],recright[10],rectop[10],recbottom[10];
   int k;

  for(int i = 0;i < point.size();i++)
   {
       k=(int)point[i].x/64;
    //   std::cout<<" x:"<<point[i].x<<" k:"<<k;
       Num[k]++;
       Sum[k]+=point[i].x;
    }
   for(k=0; k<10; k++){
       if(Num[k]!=0)
       {
       middle[round_1_Num]=Sum[k]/Num[k];
       std::cout<<"   middle:"<<middle[round_1_Num];
       round_1_Num++;
       }
   }
   std::cout<<"  round_1_Num:"<<round_1_Num;
  memset(Num,0,sizeof(Num));
  memset(Sum,0,sizeof(Sum));
   for(int i = 0;i < point.size();i++)
   {
       for(k=0; k<round_1_Num; k++){
              if(point[i].x>=(middle[k]-32) && point[i].x<(middle[k]+32)){
               Num[k]++;
               Sum[k]+=point[i].x;
            }
        }
   }
   for(k=0; k<round_1_Num; k++){
       if(Num[k]>=5){
          middle[round_2_Num]=Sum[k]/Num[k];
          std::cout<<"round-1-middle:"<<middle[round_2_Num];
          round_2_Num++;
       }
   }
  // waitKey();
 //  近的类合并

   for(k=1; k<round_2_Num; k++){
      if(abs(middle[k]-middle[k-1])<20)
             middle[k]=0;
   }


   memset(recright,0,sizeof(recright));
   memset(recbottom,0,sizeof(recbottom));
   memset(recleft,1,sizeof(recleft));
   memset(rectop,1,sizeof(rectop));
 //  for(k=0; k<10; k++){
 //      recleft[k]=640;
//       rectop[k]=320;
//   }

   memset(Num,0,sizeof(Num));
   memset(Sum,0,sizeof(Sum));

   for(int i = 0;i < point.size();i++)
   {
       for(k=0; k<round_2_Num; k++){
           if(middle[k]>0){
            if(point[i].x>=(middle[k]-52) && point[i].x<(middle[k]+52)){
               Num[k]++;
               Sum[k]+=point[i].x;
               if(point[i].x<recleft[k])
                   recleft[k]=(int)point[i].x;
               if(point[i].x>recright[k])
                   recright[k]=(int)point[i].x;
               if(point[i].y<rectop[k])
                   rectop[k]=(int)point[i].y;
               if(point[i].y> recbottom[k])
                   recbottom[k]=(int)point[i].y;

        //       std::cout<<"left"<<recleft[k]<<"recright"<<recright[k]<<"rectop"
        //               <<rectop[k]<<"recbottom"<<recbottom[k];
            }
           }
        }
   }
    for(k=0; k<round_2_Num; k++){
       if(middle[k]>0){
            middle[objNum]=Sum[k]/Num[k];
            objRect[objNum].x=recleft[k];
            objRect[objNum].y=rectop[k];
            objRect[objNum].width=recright[k]-recleft[k];
            objRect[objNum].height=recbottom[k]-rectop[k];
            std::cout<<"Rect"<<objRect[objNum].x<<" "<<objRect[objNum].y<<" "
                   <<objRect[objNum].width<<" "<<objRect[objNum].height;
            objNum++;
       }
    }
    std::cout<<"objNum:"<<objNum<<std::endl;
   return;
}
