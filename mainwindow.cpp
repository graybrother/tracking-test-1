#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "sys/time.h"
#include "videoprocessor.h"
#include "harrisdetector.h"
#include "frprocess.h"
#include "vibe-background-sequential.h"
#include "hog-svm.h"
#include "objfeature.h"

//make some change for test git usage
const int alpha_slider_max = 100;
int alpha_slider;

Mat src1;
Mat src2;
Mat dst;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_openimage_button_clicked()
{
   filename = QFileDialog::getOpenFileName(this, tr("select image file"), ".",
                                              tr("Image Files(*.jpg *.jpeg *.bmp *.png)"));

    if (filename.isEmpty()) {
        ui->warning_lineEdit->setText("file open does not success !!!");
        return;
    }
    ui->lineEdit->setText(filename);
    ui->warning_lineEdit->clear();
    ui->pushButton->setEnabled(true);
    return;

}

void MainWindow::on_pushButton_clicked()
{
    flip(image,result,1);
    cvtColor(result,result,CV_RGB2BGR);
    QImage img = QImage((const unsigned char*)(result.data), result.cols, result.rows, QImage::Format_RGB888);
 //   ui->showlabel->setPixmap(QPixmap::fromImage(img));

}


void MainWindow::on_openvideo_button_clicked()
{
    filename = QFileDialog::getOpenFileName(this, tr("select video file"), ".",
                                              tr("Vedio Files(*.mp4 *.avi *.mkv *.yuv)"));

    if (filename.isEmpty()) {
        ui->warning_lineEdit->setText("file open does not success !!!");
        return;
    }
    ui->lineEdit->setText(filename);
    ui->warning_lineEdit->clear();
    ui->pushButton->setEnabled(true);
    //findout codec FOURCC
/*    VideoProcessor processor;
    processor.setInput(filename.toStdString());
    char c[4];
    std::cout<<"vodeo codec int value:"<<processor.getCodec(c)<<std::endl;
    std::cout<<"video codec:"<<c[0]<<c[1]<<c[2]<<c[3]<<std::endl;
*/
     return;

}


void MainWindow::on_vp_button_clicked()
{
    VideoProcessor processor;
    //open video file

    if (filename.isEmpty())
        return;
    processor.setInput(filename.toStdString());
    char codec[4];
    processor.getCodec(codec);
    std::cout << "Codec: " << codec[0] << codec[1]
    << codec[2] << codec[3] << std::endl;
    //declare a window to display the video
    processor.displayInput("Current Frame");
    processor.displayOutput("Output Frame");
    //play the video at the original frame rate
    processor.setDelay(1000/processor.getFrameRate());
   // processor.stopAtFrameNo(100);
    //set the frame processor callback function
    //processor.dontCallProcess();
    processor.setFrameProcessor(canny);
    //set to save output to disk
   // processor.setOutput("files/teacher",".jpg",3,0);
   // processor.setOutput("bbb.mp4",0,0.0,true);
    //start the process
    processor.run();

}

void MainWindow::on_pushButton_2_clicked()
{
    int value = 50;
    int value2 = 6;


     namedWindow("main1",WINDOW_AUTOSIZE | CV_GUI_EXPANDED);
     namedWindow("main2",WINDOW_AUTOSIZE | CV_GUI_EXPANDED);
     createTrackbar( "track1", "main1", &value, 255,  NULL);

     String nameb1 = "button1";
     String nameb2 = "button2";

       createTrackbar( "track2", "main2", &value2, 255, NULL);
       setTrackbarMin("track2", "main2",5);

   //  setMouseCallback( "main2",on_mouse,NULL );

     Mat img1 = imread("files/flower.jpg");
     VideoCapture video;
     video.open("files/test.mp4");

     Mat img2,img3;

     src1 = imread("files/1.jpg");
     src2 = imread("files/2.jpg");

        if( src1.empty() ) { printf("Error loading src1 \n"); }
        if( src2.empty() ) { printf("Error loading src2 \n"); }

        alpha_slider = 0;

        namedWindow("Linear Blend", WINDOW_AUTOSIZE); // Create Window

        char TrackbarName[50];
        sprintf( TrackbarName, "Alpha x %d", alpha_slider_max );
        createTrackbar( TrackbarName, "Linear Blend", &alpha_slider, alpha_slider_max, on_trackbar );

        on_trackbar( alpha_slider, 0 );


     while( waitKey(1000/value2) != 27 )
     {
         img1.convertTo(img2,-1,1,value);
         video >> img3;

         imshow("main1",img2);
         imshow("main2",img3);
     }

     destroyAllWindows();

}

void on_trackbar( int, void* )
{
   double alpha = (double) alpha_slider/alpha_slider_max ;

   double beta = ( 1.0 - alpha );

   addWeighted( src1, alpha, src2, beta, 0.0, dst);

   imshow( "Linear Blend", dst );
}



void MainWindow::on_ffilldemo_button_clicked()
{
     ffilldemo();
}



void MainWindow::on_track_button_clicked()
{

    //make sure image file has been selected
     if (filename.isEmpty()){
         ui->warning_lineEdit->setText("File open error,please open video file first");
         return;
     }
    // Create video procesor instance
    VideoProcessor processor;
    // Create feature tracker instance
    FeatureTracker tracker;
    // Open video file
    processor.setInput(filename.toStdString());
    // set frame processor
    processor.setFrameProcessor(&tracker);
    // Declare a window to display the video
    processor.displayInput("Current Frame");
    processor.displayOutput("Tracked Features");
    // Play the video at the original frame rate
    processor.setDelay(1000/processor.getFrameRate());
    //codec value
   // char c[4];
   // processor.getCodec(c);
   // std::cout<<"vodeo codec int value:"<<processor.getCodec(c)<<std::endl;
   // std::cout<<"video codec:"<<c[0]<<c[1]<<c[2]<<c[3]<<std::endl;
    //write output video file
  //  if(!processor.setOutput("files/track.mkv")){
   //     std::cout<<"open output file error"<<std::endl;
  //      return;
  //  }
    // Start the process
    processor.run();
}

void MainWindow::on_gbsegment_button_clicked()
{
    filename = QFileDialog::getOpenFileName(this, tr("select video file"), ".",
                                          tr("video Files(*.*)"));

    if (filename.isEmpty())
    return;
    // Create video procesor instance
    VideoProcessor processor;
    // Create feature tracker instance
    BGFGSegmentor segmenter;
    segmenter.setLearnRate(0.9);
    // Open video file
    processor.setInput(filename.toStdString());
    // set frame processor
    processor.setFrameProcessor(&segmenter);
    // Declare a window to display the video
    processor.displayInput("Current Frame");
    processor.displayOutput("Segmented Frame");
    // Play the video at the original frame rate
    processor.setDelay(1000./processor.getFrameRate());
    // Start the process
    processor.run();
}

void MainWindow::on_harris_button_clicked()
{

    // Create Harris detector instance
    HarrisDetector harris;
    // Detect Harris corners
    std::vector<cv::Point> pts;

    if (filename.isEmpty())
        return;

    image = imread(filename.toStdString());       // 加载图片
    namedWindow("Original Image");
    namedWindow("Output Image");
    imshow("Original Image", image);

   // image.convertTo(result,CV_RGB2GRAY);
      cvtColor(image,result,CV_BGR2GRAY);

    // Compute Harris values
    harris.detect(result);
    harris.getCorners(pts,0.01);
    // Draw Harris corners
    harris.drawOnImage(result,pts);
    imshow("Output Image", result);
    waitKey();
    destroyAllWindows();

}

void MainWindow::on_vibe_button_clicked()
{
        using namespace std;
        static int frameNumber = 1; /* The current frame number */
        Mat frame;                  /* Current frame. */
        Mat segmentationMap;        /* Will contain the segmentation map. This is the binary output map. */
        int keyboard = 0;           /* Input from keyboard. Used to stop the program. Enter 'q' to quit. */

        filename = QFileDialog::getOpenFileName(this, tr("select Video file"), ".",
                                                  tr("Video Files(*.*)"));

        if (filename.isEmpty()) {
            return;
        }
        VideoCapture capture(filename.toStdString());
        if(!capture.isOpened())
            return ;

        namedWindow("Frame");
        namedWindow("Segmentation by ViBe");
        /* Model for ViBe. */
        vibeModel_Sequential_t *model = NULL; /* Model used by ViBe. */

        while ((char)keyboard != 'q' && (char)keyboard != 27) {
          /* Read the current frame. */
          if (!capture.read(frame)) {
            cerr << "Unable to read next frame." << endl;
            cerr << "Exiting..." << endl;
            exit(EXIT_FAILURE);
          }

          if ((frameNumber % 100) == 0) { cout << "Frame number = " << frameNumber << endl; }

          /* Applying ViBe.
           * If you want to use the grayscale version of ViBe (which is much faster!):
           * (1) remplace C3R by C1R in this file.
           * (2) uncomment the next line (cvtColor).
           */
          /* cvtColor(frame, frame, CV_BGR2GRAY); */

          if (frameNumber == 1) {
            segmentationMap = Mat(frame.rows, frame.cols, CV_8UC1);
            model = (vibeModel_Sequential_t*)libvibeModel_Sequential_New();
            libvibeModel_Sequential_AllocInit_8u_C3R(model, frame.data, frame.cols, frame.rows);
          }

          /* ViBe: Segmentation and updating. */
          libvibeModel_Sequential_Segmentation_8u_C3R(model, frame.data, segmentationMap.data);
          libvibeModel_Sequential_Update_8u_C3R(model, frame.data, segmentationMap.data);

          /* Post-processes the segmentation map. This step is not compulsory.
             Note that we strongly recommend to use post-processing filters, as they
             always smooth the segmentation map. For example, the post-processing filter
             used for the Change Detection dataset (see http://www.changedetection.net/ )
             is a 5x5 median filter. */
         // medianBlur(segmentationMap, segmentationMap, 3); /* 3x3 median filtering */

          erode(segmentationMap,segmentationMap,Mat());
        //  erode(segmentationMap,segmentationMap,Mat());
          dilate(segmentationMap,segmentationMap,Mat());
           dilate(segmentationMap,segmentationMap,Mat());
          /* Shows the current frame and the segmentation map. */
          imshow("Frame", frame);
          imshow("Segmentation by ViBe", segmentationMap);

          ++frameNumber;

          /* Gets the input from the keyboard. */
          keyboard = waitKey(33);
        }

        /* Delete capture object. */
        capture.release();

        /* Frees the model. */
        libvibeModel_Sequential_Free(model);
        destroyAllWindows();

}

void MainWindow::on_vibegray_button_clicked()
{
    using namespace std;
    static int frameNumber = 1; /* The current frame number */
    Mat inputFrame;
    Mat frame;                  /* Current gray frame. */
    Mat segmentationMap;        /* Will contain the segmentation map. This is the binary output map. */
    int keyboard = 0;           /* Input from keyboard. Used to stop the program. Enter 'q' to quit. */

    Mat element(5,5,CV_8U,cv::Scalar(1));

    vector<vector<Point> > contours;
    vector<Rect> rects;
    Rect rectemp;




    filename = QFileDialog::getOpenFileName(this, tr("select Video file"), ".",
                                              tr("Video Files(*.mp4 *.avi *.mkv *.yuv)"));

    if (filename.isEmpty()) {
        return;
    }
    VideoCapture capture(filename.toStdString());
    if(!capture.isOpened())
        return ;

    namedWindow("Frame");
    namedWindow("before dilate");
    namedWindow("Segmentation by ViBe");
    /* Model for ViBe. */
    vibeModel_Sequential_t *model = NULL; /* Model used by ViBe. */

    while ((char)keyboard != 'q' && (char)keyboard != 27) {
      /* Read the current frame. */
      if (!capture.read(inputFrame)) {
        cerr << "Unable to read next frame." << endl;
        cerr << "Exiting..." << endl;
        exit(EXIT_FAILURE);
      }

      if ((frameNumber % 100) == 0) { cout << "Frame number = " << frameNumber << endl; }

      /* Applying ViBe.
       * If you want to use the grayscale version of ViBe (which is much faster!):
       * (1) remplace C3R by C1R in this file.
       * (2) uncomment the next line (cvtColor).
       */
      //show the current frame
   //   imshow("Frame", inputFrame);
      //convert to gray
      cvtColor(inputFrame, frame, CV_BGR2GRAY);

      if (frameNumber == 1) {
        segmentationMap = Mat(frame.rows, frame.cols, CV_8UC1);
        model = (vibeModel_Sequential_t*)libvibeModel_Sequential_New();
        libvibeModel_Sequential_AllocInit_8u_C1R(model, frame.data, frame.cols, frame.rows);
      }

      /* ViBe: Segmentation and updating. */
      libvibeModel_Sequential_Segmentation_8u_C1R(model, frame.data, segmentationMap.data);
      libvibeModel_Sequential_Update_8u_C1R(model, frame.data, segmentationMap.data);

      /* Post-processes the segmentation map. This step is not compulsory.
         Note that we strongly recommend to use post-processing filters, as they
         always smooth the segmentation map. For example, the post-processing filter
         used for the Change Detection dataset (see http://www.changedetection.net/ )
         is a 5x5 median filter. */
   //   medianBlur(segmentationMap, segmentationMap, 5); /* 5x5 median filtering */

      cv::imshow("before dilate",segmentationMap);
    //  for(int i=0; i<4; i++){


      cv::erode(segmentationMap,segmentationMap,cv::Mat());  //3x3
      cv::dilate(segmentationMap,segmentationMap,element,Point(-1,-1),7);  //5x5
    //   cv::erode(segmentationMap,segmentationMap,element,Point(-1,-1),1);
      cv::erode(segmentationMap,segmentationMap,cv::Mat(),cv::Point(-1,-1),7);//3x3
      //Extract the contours so that
  //      }
      cv::findContours( segmentationMap, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

  //    contours.resize(contours0.size());
      for( size_t k = 0; k < contours.size(); k++ ){
         rectemp=boundingRect(contours[k]);
         if(rectemp.area()>1200 && rectemp.width>15 && rectemp.height>50){
             rects.push_back(rectemp);
            cv::rectangle(inputFrame, rectemp, cv::Scalar(255, 255, 255), 2);
         }
      }

   //     cv::rectangle(segmentationMap,Rect(10,10,30,30),cv::Scalar(255, 255, 255), 2);
//      cvtColor(segmentationMap,output,CV_GRAY2RGB);


      /* Shows  the segmentation map. */
      imshow("Frame", inputFrame);
      imshow("Segmentation by ViBe", segmentationMap);


      ++frameNumber;

      /* Gets the input from the keyboard. */
      keyboard = waitKey(30);
      if ((frameNumber % 100)==0){
           libvibeModel_Sequential_PrintParameters(model);
      }
    }

    /* Delete capture object. */
    capture.release();

    /* Frees the model. */
    libvibeModel_Sequential_Free(model);
    destroyAllWindows();

}

void MainWindow::on_difcanny3_button_clicked()
{
    using namespace std;
    static int framenumber = 0; /* The current frame number */

    Mat edge1;
    Mat edge2;
    Mat edge3;
    Mat diff1;
    Mat diff2;

    int keyboard = 0;           /* Input from keyboard. Used to stop the program. Enter 'q' to quit. */
    double start_time;
    double finish_time;

    filename = QFileDialog::getOpenFileName(this, tr("select Video file"), ".",
                                              tr("Video Files(*.*)"));

    if (filename.isEmpty()) {
        return;
    }
    VideoCapture capture(filename.toStdString());
    if(!capture.isOpened())
        return ;

    namedWindow("Current Frame");
    namedWindow("Segmentation by difcanny");
    namedWindow("diff1");
    namedWindow("diff2");

    while ((char)keyboard != 'q' && (char)keyboard != 27) {
      /* Read the current frame. */
      if (!capture.read(image)) {
        cerr << "Unable to read next frame." << endl;
        cerr << "Exiting..." << endl;
        exit(EXIT_FAILURE);
      }

      imshow("Current Frame",image);

      start_time=(double)clock();
      if(framenumber>1000000000)
            framenumber=100;
      framenumber+=1;
      if(framenumber==1) {
               canny(image,edge1);
      }
      if(framenumber==6)
             canny(image,edge2);

      if(framenumber==11)

             canny(image,edge3);
      if(framenumber>11 && (framenumber % 5)==1)
          {
                      edge2.copyTo(edge1);
                      edge3.copyTo(edge2);
                      canny(image,edge3);

                  absdiff(edge2,edge1,diff1);
                 // medianBlur(diff1,diff1,3);
                  absdiff(edge3,edge2,diff2);
                //  medianBlur(diff2,diff2,3);

                  imshow("diff1",diff1);
                  imshow("diff2",diff2);


                  bitwise_and(diff1,diff2,result);

                  finish_time=(double)clock();
                  cout<<finish_time-start_time<<"ms"<<framenumber<<"framenum"<<endl;

                 imshow("Segmentation by difcanny",result);
      }
              keyboard= waitKey(33);

    }   //end of while( keyboard

            capture.release();
            destroyAllWindows();
}


// compute the Local Binary Patterns of a gray-level image
void lbp(const cv::Mat &image, cv::Mat &result) {

    assert(image.channels() == 1); // input image must be gray scale

    result.create(image.size(), CV_8U); // allocate if necessary

    for (int j = 1; j<image.rows - 1; j++) { // for all rows (except first and last)

        const uchar* previous = image.ptr<const uchar>(j - 1); // previous row
        const uchar* current  = image.ptr<const uchar>(j);	   // current row
        const uchar* next     = image.ptr<const uchar>(j + 1); // next row

        uchar* output = result.ptr<uchar>(j);	// output row

        for (int i = 1; i<image.cols - 1; i++) {

            // compose local binary pattern
            *output =  previous[i - 1] > current[i] ? 1 : 0;
            *output |= previous[i] > current[i] ?     2 : 0;
            *output |= previous[i + 1] > current[i] ? 4 : 0;

            *output |= current[i - 1] > current[i] ?  8 : 0;
            *output |= current[i + 1] > current[i] ? 16 : 0;

            *output |= next[i - 1] > current[i] ?    32 : 0;
            *output |= next[i] > current[i] ?        64 : 0;
            *output |= next[i + 1] > current[i] ?   128 : 0;

            output++; // next pixel
        }
    }

    // Set the unprocess pixels to 0
    result.row(0).setTo(cv::Scalar(0));
    result.row(result.rows - 1).setTo(cv::Scalar(0));
    result.col(0).setTo(cv::Scalar(0));
    result.col(result.cols - 1).setTo(cv::Scalar(0));
}

void MainWindow::on_lbpface_button_clicked()
{
    cv::Mat image = imread("girl.jpg", cv::IMREAD_GRAYSCALE);

    cv::imshow("Original image", image);

    cv::Mat lbpImage;
    lbp(image, lbpImage);

    cv::imshow("LBP image", lbpImage);

    cv::Ptr<cv::face::FaceRecognizer> recognizer =
        cv::face::createLBPHFaceRecognizer(1,      // radius of LBP pattern
                                           8,      // the number of neighboring pixels to consider
                                           8, 8,   // grid size
                                           200.);  // minimum distance to nearest neighbor

    // vectors of reference image and their labels
    std::vector<cv::Mat> referenceImages;
    std::vector<int> labels;
    // open the reference images
    referenceImages.push_back(cv::imread("face0_1.png", cv::IMREAD_GRAYSCALE));
    labels.push_back(0); // person 0
    referenceImages.push_back(cv::imread("face0_2.png", cv::IMREAD_GRAYSCALE));
    labels.push_back(0); // person 0
    referenceImages.push_back(cv::imread("face1_1.png", cv::IMREAD_GRAYSCALE));
    labels.push_back(1); // person 1
    referenceImages.push_back(cv::imread("face1_2.png", cv::IMREAD_GRAYSCALE));
    labels.push_back(1); // person 1

    // the 4 positive samples
    cv::Mat faceImages(2 * referenceImages[0].rows, 2 * referenceImages[0].cols, CV_8U);
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++) {

            referenceImages[i * 2 + j].copyTo(faceImages(cv::Rect(j*referenceImages[i * 2 + j].cols, i*referenceImages[i * 2 + j].rows, referenceImages[i * 2 + j].cols, referenceImages[i * 2 + j].rows)));
        }

    cv::resize(faceImages, faceImages, cv::Size(), 0.5, 0.5);
    cv::imshow("Reference faces", faceImages);

    // train the recognizer by
    // computing the LBPHs
    recognizer->train(referenceImages, labels);

    int predictedLabel = -1;
    double confidence = 0.0;

    // Extract a face image
    cv::Mat inputImage;
    cv::resize(image(cv::Rect(160, 75, 90, 90)), inputImage, cv::Size(256, 256));
    cv::imshow("Input image", inputImage);

    // predict the label of this image
    recognizer->predict(inputImage,     // face image
                        predictedLabel, // predicted label of this image
                        confidence);    // confidence of the prediction

    std::cout << "Image label= " << predictedLabel << " (" << confidence << ")" << std::endl;

    cv::waitKey();
    destroyAllWindows();
}


void MainWindow::on_haar_button_clicked()
{
    // open the positive sample images
    std::vector<cv::Mat> referenceImages;
    referenceImages.push_back(cv::imread("stopSamples/stop00.png"));
    referenceImages.push_back(cv::imread("stopSamples/stop01.png"));
    referenceImages.push_back(cv::imread("stopSamples/stop02.png"));
    referenceImages.push_back(cv::imread("stopSamples/stop03.png"));
    referenceImages.push_back(cv::imread("stopSamples/stop04.png"));
    referenceImages.push_back(cv::imread("stopSamples/stop05.png"));
    referenceImages.push_back(cv::imread("stopSamples/stop06.png"));
    referenceImages.push_back(cv::imread("stopSamples/stop07.png"));

    unsigned int i,j;
    // create a composite image
    cv::Mat positveImages(2 * referenceImages[0].rows, 4 * referenceImages[0].cols, CV_8UC3);
    for (i = 0; i < 2; i++)
        for (j = 0; j < 4; j++) {

            referenceImages[i * 2 + j].copyTo(positveImages(cv::Rect(j*referenceImages[i * 2 + j].cols, i*referenceImages[i * 2 + j].rows, referenceImages[i * 2 + j].cols, referenceImages[i * 2 + j].rows)));
        }

    cv::imshow("Positive samples", positveImages);

    cv::Mat negative = cv::imread("stopSamples/bg01.jpg");
    cv::resize(negative, negative, cv::Size(), 0.33, 0.33);
    cv::imshow("One negative sample", negative);

    cv::Mat inputImage = cv::imread("stopSamples/stop0.jpg");
    cv::resize(inputImage, inputImage, cv::Size(), 0.5, 0.5);

    cv::CascadeClassifier cascade;
    if (!cascade.load("stopSamples/classifier/cascade.xml")) {
        std::cout << "Error when loading the cascade classfier!" << std::endl;
        return ;
    }

    // predict the label of this image
    std::vector<cv::Rect> detections;

    cascade.detectMultiScale(inputImage, // input image
                             detections, // detection results
                             1.1,        // scale reduction factor
                             1,          // number of required neighbor detections
                             0,          // flags (not used)
                             cv::Size(48, 48),    // minimum object size to be detected
                             cv::Size(128, 128)); // maximum object size to be detected

    std::cout << "detections= " << detections.size() << std::endl;
    for (i = 0; i < detections.size(); i++)
        cv::rectangle(inputImage, detections[i], cv::Scalar(255, 255, 255), 2);

    cv::imshow("Stop sign detection", inputImage);
    waitKey();

    // Detecting faces
    cv::Mat picture = cv::imread("teacher.jpg");
    cv::CascadeClassifier faceCascade;
    if (!faceCascade.load("haarcascades/haarcascade_frontalface_alt.xml")) {
        std::cout << "Error when loading the face cascade classfier!" << std::endl;
        return ;
    }

    faceCascade.detectMultiScale(picture, // input image
        detections, // detection results
        1.1,        // scale reduction factor
        3,          // number of required neighbor detections
        0,          // flags (not used)
        cv::Size(24, 24),    // minimum object size to be detected
        cv::Size(128, 128)); // maximum object size to be detected

    std::cout << " face detections= " << detections.size() << std::endl;
    // draw detections on image
    for (i = 0; i < detections.size(); i++)
        cv::rectangle(picture, detections[i], cv::Scalar(255, 255, 255), 2);

    waitKey();

    // Detecting eyes
    cv::CascadeClassifier eyeCascade;
    if (!eyeCascade.load("haarcascades/haarcascade_eye.xml")) {
        std::cout << "Error when loading the eye cascade classfier!" << std::endl;
        return ;
    }

    eyeCascade.detectMultiScale(picture, // input image
        detections, // detection results
        1.1,        // scale reduction factor
        3,          // number of required neighbor detections
        0,          // flags (not used)
        cv::Size(6, 6),    // minimum object size to be detected
        cv::Size(64, 64)); // maximum object size to be detected

    std::cout << "eye detections= " << detections.size() << std::endl;
    // draw detections on image
    for (i = 0; i < detections.size(); i++)
        cv::rectangle(picture, detections[i], cv::Scalar(0, 0, 0), 2);

    cv::imshow("Detection results", picture);

    cv::waitKey();
    destroyAllWindows();
}

void MainWindow::on_upperbody_button_clicked()
{
    VideoProcessor processor;
    HaarCascadeTracker casCadeTracker;

    if (filename.isEmpty())
        return;

    processor.setInput(filename.toStdString());

    //declare a window to display the video
    processor.displayInput("Input Frame");
    processor.displayOutput("Output Frame");
    //play the video at the original frame rate
    processor.setDelay(100/processor.getFrameRate());
   // processor.stopAtFrameNo(100);
    //set the frame processor callback function
    //processor.dontCallProcess();
    processor.setFrameProcessor(&casCadeTracker);

    processor.run();
}

void MainWindow::on_hog_button_clicked()
{
   //make sure image file has been selected
    if (filename.isEmpty()){
        ui->warning_lineEdit->setText("File open error,please open image file first");
        return;
    }

    image = imread(filename.toStdString(), cv::IMREAD_GRAYSCALE);
    if(image.empty()){
        ui->warning_lineEdit->setText("Image open error,please open correct image file");
        return;
    }
    cv::imshow("Original image", image);

    cv::HOGDescriptor hog(cv::Size((image.cols / 16) * 16, (image.rows / 16) * 16), // size of the window
        cv::Size(16, 16),    // block size
        cv::Size(16, 16),    // block stride
        cv::Size(4, 4),      // cell size
        9);                  // number of bins

    std::vector<float> descriptors;

    // Draw a representation of HOG cells
    cv::Mat hogImage= image.clone();
    drawHOGDescriptors(image, hogImage, cv::Size(16, 16), 9);
    cv::imshow("HOG image", hogImage);
    waitKey();
    destroyAllWindows();
    return;
}

void MainWindow::on_hogpeople_button_clicked()
{
    // People detection
    if (filename.isEmpty()){
        ui->warning_lineEdit->setText("File open error,please open image file first");
        return;
    }

    cv::Mat myImage = imread(filename.toStdString(), cv::IMREAD_GRAYSCALE);

    // create the detector
    std::vector<cv::Rect> peoples;
    cv::HOGDescriptor peopleHog;
    peopleHog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
    // detect peoples oin an image
    peopleHog.detectMultiScale(myImage, // input image
        peoples, // ouput list of bounding boxes
        0,       // threshold to consider a detection to be positive
        cv::Size(4, 4),   // window stride
        cv::Size(32, 32), // image padding
        1.1,              // scale factor
        2);               // grouping threshold (0 means no grouping)

    // draw detections on image
    std::cout << "Number of peoples detected: " << peoples.size() << std::endl;
    for (int i = 0; i < peoples.size(); i++)
        cv::rectangle(myImage, peoples[i], cv::Scalar(255, 255, 255), 2);

    cv::imshow("People detection", myImage);

    cv::waitKey();
    destroyAllWindows();
}

void MainWindow::on_yuv420sp_button_clicked()
{
    int nRows=480, nCols=832;
    int i,j;
    int frameNum=0;
    uchar* p;
    uchar* p1;
    uchar* temp;
    FILE* pFp;
    Mat output;

    Mat frame(nRows+nRows/2,nCols,CV_8UC1);
 //   cvtColor(frame, frame, CV_YUV2BGR_NV21);
    temp=(uchar*)malloc(nRows*nCols/2);

    namedWindow("Frame");

    //make sure image file has been selected
     if (filename.isEmpty()){
         ui->warning_lineEdit->setText("File open error,please open image file first");
         return;
     }

      pFp = fopen(filename.toStdString().data() ,"rb");
      if (feof(pFp))
      {
          std::cout<<"end of file!"<<std::endl;
          fseek(pFp, 0 , SEEK_SET);
      }

    //  p = I.ptr<uchar>(i);
      p=frame.data;

      while(!feof(pFp)){
      if ( 1 != fread(p, nRows*nCols*3/2, 1, pFp))
             break;

/*
      p=frame.data+nRows*nCols;
      if ( 1 != fread(p, nRows*nCols/2, 1, pFp))
            {
         ui->warning_lineEdit->setText("Image read error--UV!");
         return;
            }
*/
 /*
      if ( 1 != fread(temp, nRows*nCols/2, 1, pFp))
            {
         ui->warning_lineEdit->setText("Image read error--UV!");
         return;
            }
     p=frame.data+nRows*nCols;
     p1=p+nRows*nCols/4;
     for(i=0; i< nRows*nCols/2; i+=2){
         *(p+i/2)=*(temp+i);
         *(p1+i/2)=*(temp+i+1);
     }
*/
 //    cv::cvtColor(frame,frame,CV_YUV420sp2BGR);
     cv::cvtColor(frame,output,CV_YUV2BGRA_I420);
     cv::imshow("Frame", output);

     waitKey(10);

    }
    ui->warning_lineEdit->setText("End of file!");
    fclose(pFp);

//    cv::waitKey();
    destroyAllWindows();
    return;
}

void MainWindow::on_equa_button_clicked()
{
    Mat frame;
    Mat gray;
    int keyboard = 0;
    //make sure image file has been selected
     if (filename.isEmpty()){
         ui->warning_lineEdit->setText("File open error,please open video file first");
         return;
              }
     VideoCapture capture(filename.toStdString());
     if(!capture.isOpened())
         return ;

     namedWindow("Original");
     namedWindow("Effect");
     namedWindow("GaussionBlur");

     while ((char)keyboard != 'q' && (char)keyboard != 27) {
       /* Read the current frame. */
       if (!capture.read(frame)) {
         std::cout << "Unable to read next frame." << std::endl;
         std::cout << "Exiting..." << std::endl;
         //exit(EXIT_FAILURE);
         break;
       }

       cvtColor(frame,gray,CV_RGB2GRAY);
       imshow("Original",gray);

       equalizeHist(gray,gray);

      imshow("Effect",gray);

       GaussianBlur(gray,gray,Size(5,5),0,0);
       imshow("GaussionBlur",gray);
       keyboard=waitKey(33);
     }
     capture.release();

     destroyAllWindows();
     return;

}

void MainWindow::on_lkvb_button_clicked()
{
   //     int TrustThres= 3;
   //     int maxObjNum=  4;
        using namespace std;

        int left=80;
        int top=80;
        int width=540;
        int height=120;
        int CENTERx=width/2;
        int CENTERy=height/2;
        Rect processRange(left,top,width,height);

        bool haveScreenRange=true;
        int screenLeft=220-left;
        int screenTop=100-top;
        int screenWidth=125;
        int screenHeight=78;
        Rect screenRange(screenLeft,screenTop,screenWidth,screenHeight);


        static int frameNumber = 1; /* The current frame number */
        Mat inputFrame;
        Mat colorFrame;
        Mat frame;                  /* Current gray frame. */
        Mat frame_prev; // previous gray-level image
        Mat segmentationMap;        /* Will contain the segmentation map. This is the binary output map. */
        Mat updateMap;
        int keyboard = 0;           /* Input from keyboard. Used to stop the program. Enter 'q' to quit. */
        int k;
        int i,j,index,disToCenter;
        Mat element(5,5,CV_8U,cv::Scalar(1));

        vector<vector<Point> > contours;
        vector<Rect> rects;
        vector<Rect> recNoSort;

        Rect rectemp;
        Rect recpre;

 //       Rect maxrect;
        int rectsNum=0;

        Point2f p;
        uint8_t *segTemp;

        int matched[MAXOBJNUM];
        // tracked features from 0->1
    //    std::vector<cv::Point2f> points[2];
        // initial position of tracked points
    //    std::vector<cv::Point2f> initial;
     //   std::vector<cv::Point2f> grid; // created features

        std::vector<uchar> status; // status of tracked features
        std::vector<float> err; // error in tracking

        object_Feature_t objFeature[MAXOBJNUM];
       // object_Feature_t* objTemp;
        int matchedNum= 0;
        int objNum=     0;
        int currentID=  -1;
        int objJointed=0;
        int objSplited=0;
        int jointedIndex=-1;
        int currentDirection=-1;
        int jointedDirection=-1;
        int noObjCount=0;
        bool haveMatchedRect=false;

        // output parameters
        Rect outputRect(left+CENTERx-20,top+CENTERy-20,40,40);
        int  outputTimeout=0;
        bool inScreenRange=false;

        for(i=0; i< MAXOBJNUM; i++) {
            object_Feature_Init(objFeature[i]);
        }


        filename = QFileDialog::getOpenFileName(this, tr("select Video file"), ".",
                                                  tr("Video Files(*.mp4 *.avi *.mkv *.yuv)"));

        if (filename.isEmpty()) {
            ui->warning_lineEdit->setText("Filename is empty,please open video file first");
            return;
           }
        VideoCapture capture(filename.toStdString());
        if(!capture.isOpened()){
            ui->warning_lineEdit->setText("File open error,please open video file first");
            return ;
        }
        namedWindow("Frame");
        namedWindow("Gray");
        namedWindow("updateMap");
        namedWindow("Segmentation by ViBe");
        namedWindow("Afterdilate");
        namedWindow("Tracking");
        /* Model for ViBe. */
        vibeModel_Sequential_t *model = NULL; /* Model used by ViBe. */

        while ((char)keyboard != 'q' && (char)keyboard != 27) {
          /* Read the current frame. */
          if (!capture.read(inputFrame)) {
              ui->warning_lineEdit->setText("end of File ");
              capture.release();

              /* Frees the model. */
              libvibeModel_Sequential_Free(model);
              destroyAllWindows();
              return;
          }

           Mat roiframe= inputFrame(processRange);
           roiframe.copyTo(colorFrame);
          //show the current frame
       //   imshow("Frame", inputFrame);
          //convert to gray
          cvtColor(colorFrame, frame, CV_BGR2GRAY);
          imshow("Gray",frame);

        // Applying ViBe.
          if (frameNumber == 1) {
            segmentationMap = Mat(frame.rows, frame.cols, CV_8UC1);
            model = (vibeModel_Sequential_t*)libvibeModel_Sequential_New();
            libvibeModel_Sequential_AllocInit_8u_C1R(model, frame.data, frame.cols, frame.rows);
          }

          /* ViBe: Segmentation and updating. */
          //give some time for background building
          if (frameNumber < 400){
            libvibeModel_Sequential_Segmentation_8u_C1R(model, frame.data, segmentationMap.data);
            libvibeModel_Sequential_Update_8u_C1R(model, frame.data, segmentationMap.data);
            ++frameNumber;
            continue;
          }
          if(frameNumber>10000000)
              frameNumber=500;
          ++frameNumber;

          // for first image of the sequence
          if(frame_prev.empty())
            frame.copyTo(frame_prev);

         // ViBe: Segmentation

          libvibeModel_Sequential_Segmentation_8u_C1R(model, frame.data, segmentationMap.data);

          imshow("Segmentation by ViBe",segmentationMap);
          segmentationMap.copyTo(updateMap);
          /* Post-processes the segmentation map. This step is not compulsory.
             Note that we strongly recommend to use post-processing filters, as they
             always smooth the segmentation map. For example, the post-processing filter
             used for the Change Detection dataset (see http://www.changedetection.net/ )
             is a 5x5 median filter. */
         // medianBlur(segmentationMap, segmentationMap, 5); /* 5x5 median filtering */
          cv::erode(segmentationMap,segmentationMap,cv::Mat());
          cv::dilate(segmentationMap,segmentationMap,element,Point(-1,-1),7);
          cv::erode(segmentationMap,segmentationMap,cv::Mat(),Point(-1,-1),6);
       //   cv::erode(segmentationMap,segmentationMap,cv::Mat(),cv::Point(-1,-1),3);
          //Extract the contours so that

          cv::findContours( segmentationMap, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

      //    contours.resize(contours0.size());
      //    maxrect.width=1;
      //    maxrect.height=1;
          rectsNum=0;

          recNoSort.clear();
   //       recpre=Rect(1920,1080,10,10);
          for(k = 0; k < contours.size(); k++ ){
             rectemp=boundingRect(contours[k]);
             cv::rectangle(segmentationMap, rectemp, cv::Scalar(128), 4);
             //rect is the same of screenrange,let it go
    //         if(haveScreenRange && isSameRect(rectemp,screenRange))
    //             continue;
             if(rectemp.area()>500 && rectemp.area()<50000
                     && rectemp.width>10 && rectemp.height>15
                     && rectemp.width<500 && rectemp.height<155){

                  recNoSort.push_back(rectemp);
                  rectsNum++;
             }

          } //end of for(k<contours.size()
         //no rect detected, nothing to do
          if(rectsNum==0)
          {
              if(frameNumber>400){
                  objNum=     0;
                  currentID=  -1;
                  objJointed=0;
                  objSplited=0;
                   jointedIndex=-1;
                    currentDirection=-1;
                 jointedDirection=-1;
                   noObjCount=0;

                  for(i=0; i< MAXOBJNUM; i++) {
                      object_Feature_Init(objFeature[i]);
                  }
              }
              frameNumber++;
              goto output;
          }
          //sort the rects
          BubbleSort(recNoSort,rectsNum);

          // conbine some rects
          rects.clear();
          rects.push_back(recNoSort[0]);
          recpre=rects[0];
          i=1;
          for(k=1; k<rectsNum; k++){
              rectemp=recNoSort[k];
              if(abs(rectemp.x+rectemp.width/2-recpre.x-recpre.width/2)<20){
                             // some rect splited to up and down two parts
                             // just joint them together
                  rectemp=rectemp | recpre;
                  rects.pop_back();
                  rects.push_back(rectemp);
                  recpre=rectemp;
                  cout<<"rect joined "<<recpre.x<<"   "<<rectemp.x<<endl;

               }
              else{
                  rects.push_back(rectemp);
                  recpre=rectemp;
                  i++;
              }
          }
          rectsNum=i;

          cout<<"rects   " <<rectsNum<<endl;
          for(k=0; k<rectsNum; k++){
          cv::rectangle(segmentationMap, rects[k], cv::Scalar(255), 2);
          }
          imshow("Afterdilate", segmentationMap);

     //     cv::rectangle(segmentationMap,Rect(10,10,30,30),cv::Scalar(255, 255, 255), 2);
    //      cvtColor(segmentationMap,output,CV_GRAY2RGB);

          if(rectsNum>MAXOBJNUM)
              rectsNum=MAXOBJNUM;

 //         waitKey();
          //match already matched objFeture to rects
          matchedNum=0;
          for(i=0;i<MAXOBJNUM; i++)
          {
              matched[i]=0;
          }
          //first to match the currentobj
          haveMatchedRect=false;
          if(currentID>-1){
              for(k=0; k<rectsNum; k++){
                  if(isMatchedRect(objFeature[currentID].rect,rects[k]))
                    {
                        objFeature[currentID].rectIndex=k;
                        if(objNum<2)
                            matched[k]=1;
                        matchedNum++;
                        haveMatchedRect=true;
                        cout<<"a currentobj matched  id= "<<currentID<<"  k= "<<k;
                        cout<<"rect x width  "<<rects[k].x<<" "<<rects[k].width<<endl;
                        break;
                    }
              }
              if(!haveMatchedRect){
                  objNum-=1;
                  if(objNum<0)
                      objNum=0;
                  object_Feature_Init(objFeature[currentID]);
                  cout<<"current obj removed rect= "<<objFeature[currentID].rectIndex<<endl;
                  currentID=-1;
              }
          }    //end of match currentobj

          index=0;
          if(objJointed>0){
              //if obj[jointedIndex] can match two rects
              //do not match currentobj rect again
              for(k=0; k<rectsNum; k++){
                 if(isMatchedRect(objFeature[jointedIndex].rect,rects[k]))
                              index++;
              }
              if(index>1){
                  matched[objFeature[currentID].rectIndex]=1;
                  objSplited=1;
                  cout<<"rect split-------";
              }
          }
          //matching other objects
          for(i=0; i<MAXOBJNUM; i++)
          {
             cout<<"objFeature[]= "<<i;
             if(objFeature[i].rectIndex >= 0 && !objFeature[i].isCurrentObj)
             {

                 haveMatchedRect=false;
                 for(k=0; k<rectsNum; k++)
                 {
                    if(isMatchedRect(objFeature[i].rect,rects[k])
                            && matched[k]==0)
                      {
                     //     if(objJointed==1 && k==objFeature[currentID].rectIndex)
                     //         continue;             //jointed and same as currentobj,not match
                          objFeature[i].rectIndex=k;
                      //    objFeature[i].rect=rects[k];
                          matched[k]=1;
                          matchedNum++;
                          haveMatchedRect=true;
                          cout<<"  has a match  i= "<<i<<"  k= "<<k;
                          cout<<" rect x,width  "<<rects[k].x<<" "<<rects[k].width;
                      }
                  }
                 if(!haveMatchedRect ) {     //if there is a objFeture have no matched rect

                     if(objFeature[i].trustedValue==4){  //a confirmed obj removed
                         objNum-=1;
                         if(objNum<0)
                             objNum=0;
                   //      if(objFeature[i].isCurrentObj)
                   //          currentID=-1;
                     }
                     object_Feature_Init(objFeature[i]);
                     //9.20 a jointed obj removed, to not jointed status
                     if(i==jointedIndex){
                         jointedIndex=-1;
                         objJointed=0;
                     }
                     cout<<"no match, object(comfirmed or not) removed"<<i<<endl;
                 }
              }
             cout<<endl;
          }


 //         if(matchedNum<rectsNum)
 //         {
           //to treat jointed, when match currentObj,we may not set match[]
           //here add it. means this rect has been mached by currentObj
            if(currentID>=0){
                matched[objFeature[currentID].rectIndex]=1;
            }

             //if there is new rects that not matched to a obj , add to objFeture
            for(k=0; k<rectsNum; k++)
            {
                if(matched[k]==0)                   //not matched rect
                {
                    for(i=0; i<MAXOBJNUM; i++)
                    {
                     if(objFeature[i].rectIndex >=0)    //already mateched obj,continue
                        continue;
                     objFeature[i].rectIndex= k;
                     objFeature[i].rect=rects[k];
                     objFeature[i].trustedValue =1;  //a obj needed to confirm
                     matchedNum++;
                //     matched[k]=1;
                     cout<<"a rect to add"<<endl;
                     break;
                     }
                    }
                }
   //         }
          cout<<" matched number"<<matchedNum<<endl;
           //begin to do lk tracking

          for(k=0; k<MAXOBJNUM; k++)
             {
            // 1. if new feature points must be added

              if( objFeature[k].rectIndex>=0            //matched
                      && objFeature[k].trustedValue>=0      //needed tracking
                      &&(objFeature[k].points[0].size() < 100 || objSplited==1))
                                   //not enough points or objSplited
              {
                    objFeature[k].points[0].clear();
                    for (int i = 0; i < 10; i++) {
                        for (int j = 0; j < 20; j++) {
                            index=objFeature[k].rectIndex;
                            p.x=rects[index].x+i*rects[index].width/10.;
                            p.y=rects[index].y+j*rects[index].height/20.;
                            objFeature[k].points[0].push_back(p);
                        }
                    }
               }  //end of add points

              // 2. do lk track
                if(objFeature[k].rectIndex>=0   //matched
                        && objFeature[k].trustedValue>=0
                        && objFeature[k].points[0].size()>50) //need to tracking
                {
                cv::calcOpticalFlowPyrLK(
                     frame_prev, frame, // 2 consecutive images
                      objFeature[k].points[0], // input point positions in first image
                      objFeature[k].points[1], // output point positions in the 2nd image
                       status, // tracking success
                       err); // tracking error
                //  a.loop over the tracked points to reject some
                int pointNum=0;
                double xmove=0;
                double ymove=0;
                double sumXmove=0;
                double sumYmove=0;
                for( int i= 0; i < objFeature[k].points[1].size(); i++ ) {
                //  do we keep this point?
                   if(status[i]){
                       xmove=objFeature[k].points[0][i].x-objFeature[k].points[1][i].x;
                       ymove=objFeature[k].points[0][i].y-objFeature[k].points[1][i].y;
                       if (// if point has moved
                            (abs(xmove)+abs(ymove))>0.2
                            && (abs(xmove)+abs(ymove))<20){
                        //  keep this point in vector
                        objFeature[k].points[1][pointNum++] = objFeature[k].points[1][i];
                        sumXmove += xmove;
                        sumYmove += ymove;
                       }

                    }
                }
                // b. eliminate unsuccesful points
                objFeature[k].points[1].resize(pointNum);
                //  c. handle the accepted tracked points

                for (i=0; i< objFeature[k].points[1].size(); i++) {
                        cv::circle(colorFrame, objFeature[k].points[1][i], 1,
                                cv::Scalar(0, 0, 255), -1);
                }

                  //  objFeature[k].rect= getRect(objFeature[k].points[1]);
                // record movement
                objFeature[k].mvIndex++;
                if(objFeature[k].mvIndex==MOVENUM)
                        objFeature[k].mvIndex=0;
                index=objFeature[k].mvIndex;
                if(pointNum>5){
                    objFeature[k].objMvX[index]=sumXmove/pointNum;
                    objFeature[k].objMvY[index]=sumYmove/pointNum;
                }//end of if pointNum>5
                else{
                    objFeature[k].objMvX[index]=0;
                    objFeature[k].objMvY[index]=0;
                }

                // draw last frame obj and not confirmed obj
                cv::rectangle(colorFrame, objFeature[k].rect,
                              cv::Scalar(255, 255, 255), 2);
                objFeature[k].trackedPointNum=pointNum;
                objFeature[k].rect= rects[objFeature[k].rectIndex];
                objFeature[k].center.x=objFeature[k].rect.x+objFeature[k].rect.width/2;
                objFeature[k].center.y=objFeature[k].rect.y+objFeature[k].rect.height/2;
                sumXmove=0;
                sumYmove=0;
                for(i=0; i<MOVENUM; i++){
                    sumXmove+=objFeature[k].objMvX[i];
                    sumYmove+=objFeature[k].objMvY[i];
                }
                objFeature[k].moveX=sumXmove/MOVENUM;
                objFeature[k].moveY=sumYmove/MOVENUM;
                cout<<"   point number:"<<pointNum<<"  moveX:"<<objFeature[k].moveX;
                cout<<"   MvX"<<objFeature[k].objMvX[objFeature[k].mvIndex];
                cout<<"  ymove:"<<objFeature[k].moveY<<endl;

                std::swap(objFeature[k].points[1], objFeature[k].points[0]);
              }   //end of do lk tracking for one objFeature


            }  //end of do lk truacking for all obj
//       策略，
         //treat tracking status of lk
         for(k=0; k<MAXOBJNUM; k++){
             if(objFeature[k].rectIndex>=0
                     && objFeature[k].trustedValue>=0){
                 if(objFeature[k].trackedPointNum>5){   //tracking ok
                     if(objFeature[k].trustedValue<4){  //待定目标
                         objFeature[k].trustedValue+=1;
                         if(objFeature[k].trustedValue==4){
                             objFeature[k].notmoveCount=900;
                             objNum+=1;
                             cout<<"a object comfirmed index="<<k<<endl;
                         }

                     }    //end of 待定目标
                     else{
                         objFeature[k].notmoveCount=900;
                     }

                 }  //end of tracking ok
                 else{                                  //tracking not ok
                     if(objFeature[k].trustedValue<4){      //待定目标
                         objFeature[k].trustedValue-=1;
                     }
                     else{                                  //目标
                         objFeature[k].notmoveCount-=1;
                         if(objFeature[k].notmoveCount==0){
                             objNum-=1;
                             if(objNum<0)
                                 objNum=0;
                             if(objFeature[k].isCurrentObj)
                                 currentID=-1;
                             object_Feature_Init(objFeature[k]);
                             cout<<"no move timeout,a object removed"<<endl;
                         }
                     }
                 }    //end of tracking not ok
             }  //end of first if
         }  //end of for(k=0
         //treat condition that there is a rect join accured
         if(objNum>0 && currentID>=0 ){
            for(k=0; k<MAXOBJNUM; k++){
                 if(k==currentID)
                     continue;
                 //obj join happend
                 if(objFeature[k].rectIndex>-1  //9.20  trustedValue==4
                        && objFeature[k].rectIndex==objFeature[currentID].rectIndex){
                     if(objJointed==0){
                         objJointed+=1;
                         jointedIndex=k;
                         currentDirection=objFeature[currentID].moveX;
                         jointedDirection=objFeature[jointedIndex].moveX;
                         cout<<"two rect jointed"<<currentID<<"  to "<<k<<endl;

                         break;
                     }
                     else
                         objJointed+=1;
                     ui->warning_lineEdit->setText("jointed status");
                     cout<<"jointed count  "<<objJointed<<endl;
                  }
            }
          }
        //9.20
         if(objNum>1 && currentID>=0 && objJointed>200
                 && objFeature[jointedIndex].trustedValue==4){
       //  if(objNum>1 && currentID>=0 && objJointed>300
       //              && objFeature[jointedIndex].rectIndex>-1){
             //jointed so long,may be a dimatched obj,remove it

             object_Feature_Init(objFeature[jointedIndex]);
             objJointed=0;
             jointedIndex=-1;
             objNum-=1;
             ui->warning_lineEdit->setText("timeout,remove ajointed obj");
         }
         //treat condition that jointed rect split
         if(objNum>1 && currentID>=0 && objJointed>=1 && objSplited==1){
             if(objFeature[currentID].rectIndex
                     !=objFeature[jointedIndex].rectIndex){
               //rect splited
                 objJointed=0;
                 objSplited=0;
                 cout<<"rect split, choose  "<<currentDirection<<"  "<<jointedDirection;
                 cout<<"  "<<objFeature[currentID].objMvX[objFeature[currentID].mvIndex];
                 cout<<"   "<<objFeature[jointedIndex].objMvX[objFeature[jointedIndex].mvIndex];
                 cout<<"    "<<endl;

                 index=currentDirection
                         * objFeature[currentID].objMvX[objFeature[currentID].mvIndex];
                 if(objFeature[jointedIndex].trustedValue==4
                         && objFeature[jointedIndex].objMvX[objFeature[jointedIndex].mvIndex]!=0
                         && index<=0){
                     //not the same direction, exchange currentobj and joined obj
                     objFeature[currentID].isCurrentObj= false;
                     currentID= jointedIndex;
                     objFeature[currentID].isCurrentObj= true;

                 }
                 jointedIndex=-1;
                 ui->warning_lineEdit->setText("jointed obj splited");
             }
         }

       //选取currentobj 输出x,y and objNum
         disToCenter= CENTERx;
         index= -1;
         cout<<"result output ";
         if(objNum>0){              //there is at least one obj
             //there is no current obj, choose one
             if(currentID<0){
                 for(k=0; k<MAXOBJNUM; k++){
                     if(objFeature[k].trustedValue==4){     //is a confirmed obj
                         i=abs(objFeature[k].center.x-CENTERx);
                         if(i < disToCenter){   //obj near to center
                         disToCenter=i;
                         index=k;
                        }
                     }
                 }
                 currentID=index;
                 objFeature[index].isCurrentObj=true;
             }  //end if currentID<0
             //if there are at least one ScreenRange
             if(haveScreenRange){
                 //check if currentObj in screenrange
                 if(isMatchedRect(screenRange,objFeature[currentID].rect)){
                     inScreenRange=true;
                     // there are other obj, set it to currentobj
                     if(objNum>1){
                         index=-1;
                         disToCenter=CENTERx;
                         //there have some confirmed obj not in screenrange
                         for(k=0; k<MAXOBJNUM; k++){
                             if(k!=currentID
                                   && objFeature[k].trustedValue==4
                                   && !isMatchedRect(screenRange,objFeature[k].rect)){
                                 i=abs(objFeature[k].center.x-CENTERx);
                                 if(i < disToCenter){   //obj near to center
                                 disToCenter=i;
                                 index=k;
                                }
                             }
                         }
                         if(index>-1){
                             objFeature[currentID].isCurrentObj=false;
                             currentID=index;
                             objFeature[currentID].isCurrentObj=true;
                             inScreenRange=false;
                         }

                     } //end of have other obj not in screen
                  } //end of if current obj in screen
                // current obj not in screen
                 else{
                     inScreenRange=false;
                 }
             }  //end of have screenrange
         }   //end of have obj

         //output
output:
         if(currentID>-1){
             outputRect=objFeature[currentID].rect;
             outputRect.x+=left;
             outputRect.y+=top;
             outputTimeout=objFeature[currentID].notmoveCount;
         }

         cv::rectangle(inputFrame, outputRect,
                            cv::Scalar(0, 0, 255), 4);
         cout<<"currentid "<<currentID<<"  timeout="<<outputTimeout;
         cout<<"  object number="<<objNum;
         cout<<"  inScreenRange="<<inScreenRange;
         cout<<"  Jointed status  "<<objJointed<<"----------end"<<endl;
         //remove 9.21
/*         else{                      //no obj,choose one from not confirmed obj
                                    //to give out a temp result
             index=-1;
             disToCenter=CENTERx;
             for(k=0; k<MAXOBJNUM; k++){
                 if(objFeature[k].trustedValue>=0){     //is a not confirmed obj
                     i=abs(objFeature[k].center.x-CENTERx);
                     if(i < disToCenter){   //obj near to center
                         disToCenter=i;
                         index=k;
                    }
                 }
             }
             if(index>=0){          //there is a not confirmed obj
               //  currentID=index;
               //  objFeature[currentID].isCurrentObj=true;
               //  objNum++;
                 cv::rectangle(inputFrame, objFeature[index].rect,
                                    cv::Scalar(255, 0, 0), 4);
              //   cout<<"currentid "<<currentID<<"  timeout="<<objFeature[currentID].notmoveCount;
                 cout<<"  no object,a temp resultr "<<"----------end"<<endl;
             }
             else{              //realy no obj(both confirmed or not confirmed
                 cv::rectangle(inputFrame, Rect(CENTERx-25,30,50,100),
                                    cv::Scalar(0, 255, 0), 4);
                 cout<<"no object "<<endl;

                }
         }       //end of no obj
 */

         //init all where are some errors
         if(objNum>0)
             noObjCount=0;
         if(objNum==0){
             noObjCount+=1;
             if(noObjCount>5){
                 objNum=     0;
                 currentID=  -1;
                 objJointed=0;
                 objSplited=0;
                 jointedIndex=-1;
                 currentDirection=-1;
                 jointedDirection=-1;
                 noObjCount=0;

             }
         }
         //reach MAXOBJNUM,to many obj,remove them except currentobj
         if(objNum==MAXOBJNUM && currentID>-1){
             for(k=0; k<MAXOBJNUM; k++){
                 if(!objFeature[k].isCurrentObj){
                     object_Feature_Init(objFeature[k]);
                 }
             }
             objNum=1;
         }
        //set mask of current object to COLOR_CURRENTOBJ
        if(currentID>=0){
            for(i=0; i<objFeature[currentID].rect.height; i++){
                index=(i+objFeature[currentID].rect.y)* updateMap.cols;
                for(j=0; j<objFeature[currentID].rect.width; j++){
                    segTemp=updateMap.data+index
                            +objFeature[currentID].rect.x+j;
                    if(*segTemp==COLOR_FOREGROUND)
                        *segTemp=COLOR_CURRENTOBJ;
                }
            }
         }
         //VIBE background update
         libvibeModel_Sequential_Update_8u_C1R(model, frame.data, updateMap.data);

         cv::swap(frame_prev, frame);

          /* Shows  the segmentation map. */
         imshow("Frame", colorFrame);
         imshow("Tracking",inputFrame);
         imshow("updateMap",updateMap);

          /* Gets the input from the keyboard. */
          keyboard = waitKey(60);
   //       if ((frameNumber % 100)==0){
    //           libvibeModel_Sequential_PrintParameters(model);
    //           cout<<"Max rectangle: width"<< maxrect.width
    //                        <<" height"<< maxrect.height<<endl;
    //      }
    //     waitKey();

        } //end of while

        /* Delete capture object. */
        capture.release();

        /* Frees the model. */
        libvibeModel_Sequential_Free(model);
        destroyAllWindows();
        return;
}
