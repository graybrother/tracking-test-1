#-------------------------------------------------
#
# Project created by QtCreator 2017-07-10T13:48:02
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = test1
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
    videoprocessor.cpp \
    ffiledemo.cpp \
    tracker.cpp \
    bgfgsegment.cpp \
    harrisdetector.cpp \
    frprocess.cpp \
    vibe-background-sequential.c \
    haarcascade.cpp \
    hog-svm.cpp \
    objfeature.cpp

HEADERS  += mainwindow.h \
    videoprocessor.h \
    harrisdetector.h \
    frprocess.h \
    vibe-background-sequential.h \
    hog-svm.h \
    objfeature.h

FORMS    += \
    mainwindow.ui

INCLUDEPATH += c:/opencv-contrib/include/opencv \
                 c:/opencv-contrib/include/opencv2 \
                 c:/opencv-contrib/include

LIBS +=  c:/opencv-contrib/lib/libopencv_*.dll.a \
        #opencv/lib/libopencv_contrib320.dll.a \   // 该模块还在编��?
 #       /vhd_opencv/opencv/lib/libopencv_core320.dll.a \
  #      /vhd_opencv/opencv/lib/libopencv_features2d320.dll.a \
   #     /vhd_opencv/opencv/lib/libopencv_flann320.dll.a \
        #opencv/lib/libopencv_gpu320.dll.a \
    #    /vhd_opencv/opencv/lib/libopencv_highgui320.dll.a \
  #      /vhd_opencv/opencv/lib/libopencv_imgproc320.dll.a \
   #     /vhd_opencv/opencv/lib/libopencv_imgcodecs320.dll.a \
        #opencv/lib/libopencv_legacy320.dll.a \
    #    /vhd_opencv/opencv/lib/libopencv_ml320.dll.a \
     #   /vhd_opencv/opencv/lib/libopencv_objdetect320.dll.a \
     #   /vhd_opencv/opencv/lib/libopencv_video320.dll.a \
     #   /vhd_opencv/opencv/lib/libopencv_videoio320.dll.a

#DEPENDPATH += .
#MOC_DIR=temp/moc
#RCC_DIR=temp/rcc
#UI_DIR=temp/ui
#OBJECTS_DIR=temp/obj
#DESTDIR=bin */

QMAKE_CXXFLAGS += -finput-charset=UTF-8
