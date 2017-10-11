#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QFile>
#include <QFileDialog>
#include <iostream>
#include <string>

#include "opencv.hpp"
#include "face.hpp"

using namespace cv;

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private slots:
    void on_openimage_button_clicked();

    void on_pushButton_clicked();

    void on_openvideo_button_clicked();

    void on_vp_button_clicked();

    void on_pushButton_2_clicked();

    void on_ffilldemo_button_clicked();

    void on_track_button_clicked();

    void on_gbsegment_button_clicked();

    void on_harris_button_clicked();

    void on_vibe_button_clicked();

    void on_vibegray_button_clicked();

    void on_difcanny3_button_clicked();

    void on_lbpface_button_clicked();

    void on_haar_button_clicked();

    void on_upperbody_button_clicked();

    void on_hog_button_clicked();

    void on_hogpeople_button_clicked();

    void on_yuv420sp_button_clicked();

    void on_equa_button_clicked();

    void on_lkvb_button_clicked();

private:
    Ui::MainWindow *ui;
    QString filename;
    QString s;
    Mat image;
    Mat result;
};

void on_trackbar( int, void* );
int ffilldemo();

#endif // MAINWINDOW_H
