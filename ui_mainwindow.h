/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 5.3.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QScrollBar>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QWidget *centralWidget;
    QPushButton *openimage_button;
    QCheckBox *checkBox;
    QLabel *label;
    QScrollBar *horizontalScrollBar;
    QPushButton *pushButton;
    QLineEdit *lineEdit;
    QPushButton *openvideo_button;
    QPushButton *vp_button;
    QPushButton *pushButton_2;
    QPushButton *ffilldemo_button;
    QPushButton *track_button;
    QPushButton *gbsegment_button;
    QPushButton *harris_button;
    QPushButton *vibe_button;
    QPushButton *vibegray_button;
    QPushButton *difcanny3_button;
    QPushButton *lbpface_button;
    QPushButton *haar_button;
    QPushButton *upperbody_button;
    QPushButton *hog_button;
    QLineEdit *warning_lineEdit;
    QLabel *label_2;
    QPushButton *hogpeople_button;
    QPushButton *yuv420sp_button;
    QPushButton *equa_button;
    QPushButton *lkvb_button;
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QStringLiteral("MainWindow"));
        MainWindow->resize(1190, 718);
        QFont font;
        font.setFamily(QStringLiteral("Arial"));
        font.setPointSize(12);
        font.setBold(true);
        font.setWeight(75);
        MainWindow->setFont(font);
        centralWidget = new QWidget(MainWindow);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        openimage_button = new QPushButton(centralWidget);
        openimage_button->setObjectName(QStringLiteral("openimage_button"));
        openimage_button->setGeometry(QRect(630, 10, 131, 41));
        QFont font1;
        font1.setPointSize(13);
        font1.setBold(true);
        font1.setWeight(75);
        openimage_button->setFont(font1);
        openimage_button->setCursor(QCursor(Qt::ArrowCursor));
        openimage_button->setMouseTracking(false);
        openimage_button->setAutoDefault(true);
        checkBox = new QCheckBox(centralWidget);
        checkBox->setObjectName(QStringLiteral("checkBox"));
        checkBox->setGeometry(QRect(140, 620, 71, 16));
        label = new QLabel(centralWidget);
        label->setObjectName(QStringLiteral("label"));
        label->setGeometry(QRect(20, 10, 161, 31));
        horizontalScrollBar = new QScrollBar(centralWidget);
        horizontalScrollBar->setObjectName(QStringLiteral("horizontalScrollBar"));
        horizontalScrollBar->setGeometry(QRect(290, 620, 771, 16));
        horizontalScrollBar->setOrientation(Qt::Horizontal);
        pushButton = new QPushButton(centralWidget);
        pushButton->setObjectName(QStringLiteral("pushButton"));
        pushButton->setEnabled(false);
        pushButton->setGeometry(QRect(490, 520, 131, 41));
        pushButton->setFont(font1);
        lineEdit = new QLineEdit(centralWidget);
        lineEdit->setObjectName(QStringLiteral("lineEdit"));
        lineEdit->setGeometry(QRect(190, 10, 411, 31));
        QFont font2;
        font2.setPointSize(12);
        font2.setBold(false);
        font2.setWeight(50);
        lineEdit->setFont(font2);
        openvideo_button = new QPushButton(centralWidget);
        openvideo_button->setObjectName(QStringLiteral("openvideo_button"));
        openvideo_button->setGeometry(QRect(790, 10, 121, 41));
        QFont font3;
        font3.setFamily(QStringLiteral("Arial"));
        font3.setPointSize(14);
        font3.setBold(true);
        font3.setWeight(75);
        openvideo_button->setFont(font3);
        vp_button = new QPushButton(centralWidget);
        vp_button->setObjectName(QStringLiteral("vp_button"));
        vp_button->setGeometry(QRect(30, 70, 141, 41));
        vp_button->setFont(font);
        pushButton_2 = new QPushButton(centralWidget);
        pushButton_2->setObjectName(QStringLiteral("pushButton_2"));
        pushButton_2->setGeometry(QRect(50, 520, 151, 41));
        pushButton_2->setFont(font);
        ffilldemo_button = new QPushButton(centralWidget);
        ffilldemo_button->setObjectName(QStringLiteral("ffilldemo_button"));
        ffilldemo_button->setGeometry(QRect(240, 520, 211, 41));
        track_button = new QPushButton(centralWidget);
        track_button->setObjectName(QStringLiteral("track_button"));
        track_button->setGeometry(QRect(520, 70, 211, 41));
        track_button->setFont(font);
        gbsegment_button = new QPushButton(centralWidget);
        gbsegment_button->setObjectName(QStringLiteral("gbsegment_button"));
        gbsegment_button->setGeometry(QRect(770, 70, 181, 41));
        gbsegment_button->setFont(font);
        harris_button = new QPushButton(centralWidget);
        harris_button->setObjectName(QStringLiteral("harris_button"));
        harris_button->setGeometry(QRect(660, 520, 151, 41));
        harris_button->setFont(font);
        vibe_button = new QPushButton(centralWidget);
        vibe_button->setObjectName(QStringLiteral("vibe_button"));
        vibe_button->setGeometry(QRect(520, 140, 151, 41));
        vibe_button->setFont(font);
        vibegray_button = new QPushButton(centralWidget);
        vibegray_button->setObjectName(QStringLiteral("vibegray_button"));
        vibegray_button->setGeometry(QRect(770, 140, 131, 41));
        vibegray_button->setFont(font);
        difcanny3_button = new QPushButton(centralWidget);
        difcanny3_button->setObjectName(QStringLiteral("difcanny3_button"));
        difcanny3_button->setGeometry(QRect(420, 250, 251, 41));
        lbpface_button = new QPushButton(centralWidget);
        lbpface_button->setObjectName(QStringLiteral("lbpface_button"));
        lbpface_button->setGeometry(QRect(30, 250, 141, 41));
        haar_button = new QPushButton(centralWidget);
        haar_button->setObjectName(QStringLiteral("haar_button"));
        haar_button->setGeometry(QRect(210, 250, 141, 41));
        upperbody_button = new QPushButton(centralWidget);
        upperbody_button->setObjectName(QStringLiteral("upperbody_button"));
        upperbody_button->setGeometry(QRect(200, 70, 281, 41));
        hog_button = new QPushButton(centralWidget);
        hog_button->setObjectName(QStringLiteral("hog_button"));
        hog_button->setGeometry(QRect(30, 140, 141, 41));
        warning_lineEdit = new QLineEdit(centralWidget);
        warning_lineEdit->setObjectName(QStringLiteral("warning_lineEdit"));
        warning_lineEdit->setGeometry(QRect(220, 460, 491, 31));
        warning_lineEdit->setStyleSheet(QStringLiteral("color: rgb(255, 0, 0);"));
        label_2 = new QLabel(centralWidget);
        label_2->setObjectName(QStringLiteral("label_2"));
        label_2->setGeometry(QRect(120, 460, 91, 21));
        hogpeople_button = new QPushButton(centralWidget);
        hogpeople_button->setObjectName(QStringLiteral("hogpeople_button"));
        hogpeople_button->setGeometry(QRect(200, 140, 181, 41));
        yuv420sp_button = new QPushButton(centralWidget);
        yuv420sp_button->setObjectName(QStringLiteral("yuv420sp_button"));
        yuv420sp_button->setGeometry(QRect(860, 520, 161, 41));
        equa_button = new QPushButton(centralWidget);
        equa_button->setObjectName(QStringLiteral("equa_button"));
        equa_button->setGeometry(QRect(770, 250, 251, 41));
        lkvb_button = new QPushButton(centralWidget);
        lkvb_button->setObjectName(QStringLiteral("lkvb_button"));
        lkvb_button->setGeometry(QRect(30, 320, 211, 41));
        MainWindow->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(MainWindow);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 1190, 23));
        MainWindow->setMenuBar(menuBar);
        mainToolBar = new QToolBar(MainWindow);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        MainWindow->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(MainWindow);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        MainWindow->setStatusBar(statusBar);

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "MainWindow", 0));
        openimage_button->setText(QApplication::translate("MainWindow", "Open Image", 0));
        checkBox->setText(QApplication::translate("MainWindow", "CheckBox", 0));
        label->setText(QApplication::translate("MainWindow", "Opened File Name:", 0));
        pushButton->setText(QApplication::translate("MainWindow", "Process", 0));
        openvideo_button->setText(QApplication::translate("MainWindow", "Open video", 0));
        vp_button->setText(QApplication::translate("MainWindow", "VideoProcessor", 0));
        pushButton_2->setText(QApplication::translate("MainWindow", "New HighGUI", 0));
        ffilldemo_button->setText(QApplication::translate("MainWindow", "ffilldemo", 0));
        track_button->setText(QApplication::translate("MainWindow", "VideoTrack(ST-LK)", 0));
        gbsegment_button->setText(QApplication::translate("MainWindow", "BgfgSegment", 0));
        harris_button->setText(QApplication::translate("MainWindow", "HarrisDetector", 0));
        vibe_button->setText(QApplication::translate("MainWindow", "ViBe", 0));
        vibegray_button->setText(QApplication::translate("MainWindow", "ViBe Gray", 0));
        difcanny3_button->setText(QApplication::translate("MainWindow", "THREE FRAME CANNY", 0));
        lbpface_button->setText(QApplication::translate("MainWindow", "LBP Face", 0));
        haar_button->setText(QApplication::translate("MainWindow", "HAAR DECT", 0));
        upperbody_button->setText(QApplication::translate("MainWindow", "Harr+Adacascade UpperBody", 0));
        hog_button->setText(QApplication::translate("MainWindow", "DrawHog", 0));
        label_2->setText(QApplication::translate("MainWindow", "Warning:", 0));
        hogpeople_button->setText(QApplication::translate("MainWindow", "HogPeopleDetect", 0));
        yuv420sp_button->setText(QApplication::translate("MainWindow", "YUV420SP", 0));
        equa_button->setText(QApplication::translate("MainWindow", "show equalize effect", 0));
        lkvb_button->setText(QApplication::translate("MainWindow", "ST-LK+ViBe", 0));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
