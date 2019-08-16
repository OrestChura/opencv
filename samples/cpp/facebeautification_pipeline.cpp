//  This sample is an OpenCV implementation of a face beautification algorythm.
//
//  The sample uses two pretrained OpenVINO networks so the OpenVINO package has to be preinstalled.
//  Please install topologies described below using downloader.py
//  (.../openvino/deployment_tools/tools/model_downloader) to run this sample.
//  Face detection model - face-detection-adas-0001:
//  https://github.com/opencv/open_model_zoo/tree/master/intel_models/face-detection-adas-0001
//  Facial landmarks detection model - facial-landmarks-35-adas-0002:
//  https://github.com/opencv/open_model_zoo/tree/master/intel_models/facial-landmarks-35-adas-0002

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;
using namespace cv::dnn;

#include <iostream>
#include <fstream>

namespace custom{

void unsharpMask(InputArray src, OutputArray dst, int sigma, double strength)
{
    std::vector<Mat> chlsInput, chlsOut;
    split(src, chlsInput);
    size_t chls = size_t(src.channels());
    for (size_t i=0ul; i<chls; i++)
    {
        Mat medianed, lapl, tmp;
        chlsInput[i].copyTo(tmp);
        medianBlur(tmp, medianed, sigma);
        Laplacian(medianed, lapl, chlsInput[i].depth());
        tmp -= strength*lapl;
        chlsOut.push_back(tmp);
    }
    merge(chlsOut, dst);
}

void seekToLayer(std::ifstream& xml)
{
    std::string strInput;
    std::getline(xml, strInput, '<');
    std::getline(xml, strInput, '>');
    while (!(xml.eof() || strInput.compare(0, 5, "layer")))
    {
        std::getline(xml, strInput);
        std::getline(xml, strInput, '<');
        std::getline(xml, strInput, '>');
    }
}

void getNetInputParams(const std::string XmlPath, int& Cols, int& Rows, int& ObjectSize)
{
//    std::ifstream Xml(XmlPath);
//    std::vector<int> Size;
//    seekToLayer(Xml);
//    std::string StrInput;
//    if (StrInput == "dim")
//    {
//        int temp;
//        Xml >> temp;
//        Size.push_back(temp);
//    }
//    std::getline(Xml, StrInput);
//    Xml.close();
//    Rows = Size[2];
//    Cols = Size[3];
//    //WIP
    if (XmlPath == "/home/orestchura/git/opencv_extra/testdata/dnn/omz_intel_models/Transportation/object_detection/face/pruned_mobilenet_reduced_ssd_shared_weights/dldt/face-detection-adas-0001.xml")
    {
        Cols = 672;
        Rows = 384;
        ObjectSize = 7;
    }
    else
    {
        Cols = 60;
        Rows = 60;
        ObjectSize = 0;
    }
}

cv::Rect getFace(const float* faceData, const size_t imgCols, const size_t imgRows)
{
    uint faceLeft = uint(std::max(int(faceData[3] * imgCols), 0));
    uint faceTop = uint(std::max(int(faceData[4] * imgRows), 0));
    uint faceRight  = uint(std::min(int(faceData[5] * imgCols), int(imgCols - 2)));
    uint faceBot = uint(std::min(int(faceData[6] * imgRows), int(imgRows - 2)));
    uint faceWidth  = faceRight - faceLeft + 1;
    uint faceHeight = faceBot - faceTop + 1;
    return cv::Rect(faceLeft, faceTop, faceWidth, faceHeight);
}

Size getBorderSizeAddToSquare(const cv::Rect face)
{
    int faceMaxSize = std::max(face.width, face.height);
    return Size(faceMaxSize - face.width, faceMaxSize - face.height);
}

void getLandmarks(const float* landmData, Point* ptsFaceElems[18], std::vector<Point>& vctJaw)
{

}

}//namespace custom

int main(int argc, char** argv)
{
    const std::string winFB = "FaceBeautificator";
    const std::string winInput = "Input";
    namedWindow(winFB, WINDOW_NORMAL);
    namedWindow(winInput, WINDOW_NORMAL);
    const Scalar clrGreen(0, 255, 0);
    const Scalar clrYellow(0, 255, 255);
    const Scalar clrBlack(0, 0, 0);
    const double pi = 3.1415926535897;

    CommandLineParser parser(argc, argv,
     "{ help         h       ||      print the help message. }"

     "{ facepath     f       ||      full path to a Face detection model (.xml) file and weights(.bin) file directory.}"
     "{ facename             |face-detection-adas-0001|     the face detection model name.}"
     "{ facemodelext         |.xml|  the face detection model file extension.}"
     "{ faceweightsext       |.bin|  the face detection weights file extension.}"

     "{ landmpath    l       ||      full path to a facial Landmarks detection model (.xml) file and weights (.bin) file directory.}"
     "{ landmname            |facial-landmarks-35-adas-0002|     the facial landmarks detection model name.}"
     "{ landmmodelext        |.xml|  the landmarks detection model file extension.}"
     "{ landmweightsext      |.bin|  the landmarks detection weights file extension.}"

     "{ input        i       ||      full path to an input image or a video file. Skip this argument to capture frames from a camera.}"
     "{ boxes        b       |false| set true if want to draw face Boxes in the \"Input\" window.}"
     "{ landmarks    m       |false| set true if want to draw facial landMarks in the \"Input\" window.}"
     );

    parser.about("Use this script to run face beautification algorythm.");
    if (argc == 1 || parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    //Parsing input arguments
    const std::string facePath = parser.get<std::string>("facepath");
    const std::string faceName = parser.get<std::string>("facename");
    const std::string faceModelExt = parser.get<std::string>("facemodelext");
    const std::string faceWeightsExt = parser.get<std::string>("faceweightsext");
    const std::string faceXmlPath = facePath + "/" + faceName + faceModelExt;
    const std::string faceBinPath = facePath + "/" + faceName + faceWeightsExt;

    const std::string landmPath = parser.get<std::string>("landmpath");
    const std::string landmName = parser.get<std::string>("landmname");
    const std::string landmModelExt = parser.get<std::string>("landmmodelext");
    const std::string landmWeightsExt = parser.get<std::string>("landmweightsext");
    const std::string landmXmlPath = landmPath + "/" + landmName + landmModelExt;
    const std::string landmBinPath = landmPath + "/" + landmName + landmWeightsExt;

    const bool flgBoxes = parser.get<bool>("boxes");
    const bool flgLandmarks = parser.get<bool>("landmarks");

    //Models' definition & initialization
    Net faceNet = readNet(faceXmlPath, faceBinPath);
    const float faceConfThreshold = 0.7f;
    int faceCols;
    int faceRows;
    int faceObjectSize;
    custom::getNetInputParams(faceXmlPath, faceCols, faceRows, faceObjectSize);

    Net landmNet = readNet(landmXmlPath, landmBinPath);
    int landmCols;
    int landmRows;
    int landmObjectSize;
    custom::getNetInputParams(landmXmlPath, landmCols, landmRows, landmObjectSize);

    //Input
    VideoCapture cap;
    if (parser.has("input"))
    {
        cap.open(parser.get<String>("input"));
    }
    else if (!cap.open(0))
    {
        std::cout << "No input available" << std::endl;
        return 1;
    }

    Mat img;
    Mat mskNoFaces;
    Mat mskBlurs;
    Mat mskSharps;

    while (waitKey(1) < 0)
    {
        cap >> img;
        if (img.empty())
        {
           waitKey();
           break;
        }
        Mat imgDraw;
        img.copyTo(imgDraw);

        //Infering Face detector
        faceNet.setInput(blobFromImage(img, 1.0, Size(faceCols, faceRows)));
        Mat faceOut = faceNet.forward();

        mskNoFaces.create(img.rows, img.cols, CV_8UC3);
        mskNoFaces.setTo(Scalar(1, 1, 1));
        mskBlurs.create(img.rows, img.cols, CV_8UC3);
        mskBlurs.setTo(Scalar(0, 0, 0));
        mskSharps.create(img.rows, img.cols, CV_8UC3);
        mskSharps.setTo(Scalar(0, 0, 0));

        //Face boxes processing
        float* faceData = (float*)(faceOut.data);
        for (size_t i = 0ul; i < faceOut.total(); i += faceObjectSize)
        {
            float faceConfidence = faceData[i + 2];
            if (faceConfidence > faceConfThreshold)
            {
                cv::Rect face = custom::getFace(faceData + i, img.cols, img.rows);

                //Postprocessing for landmarks
                Size faceAdd = custom::getBorderSizeAddToSquare(face);
                Mat imgCrop;
                cv::copyMakeBorder(img(Rect(face.x, face.y, face.width, face.height)), imgCrop,
                                   faceAdd.height / 2, (faceAdd.height + 1) / 2,
                                   faceAdd.width / 2, (faceAdd.width + 1) / 2,
                                   BORDER_CONSTANT | BORDER_ISOLATED , cv::Scalar(0, 0, 0));

                //Infering Landmarks detector
                landmNet.setInput(blobFromImage(imgCrop, 1.0, Size(landmCols, landmRows)));
                Mat landmOut = landmNet.forward();

                //Landmarks processing
                float* landmData = (float*)(landmOut.data);
                std::vector<Point> ptsFaceElems(18);

                size_t j = 0ul;
                for (; j < 18 * 2; j += 2)
                {
                    ptsFaceElems[j / 2] = Point(int(landmData[j] * imgCrop.cols + face.x
                                                    - faceAdd.width / 2),
                                                int(landmData[j + 1] * imgCrop.rows + face.y
                                                    - faceAdd.height / 2));
                }

                std::vector<Point> vctFace;
                {
                    std::vector<Point> vctJaw;
                    vctJaw.reserve(17);
                    for(; j < landmOut.total(); j += 2)
                    {
                        vctJaw.push_back(Point(int(landmData[j] * imgCrop.cols + face.x
                                                   - faceAdd.width / 2),
                                               int(landmData[j + 1] * imgCrop.rows + face.y
                                                   - faceAdd.height / 2)));
                    }
                    Point ptJawCenter((vctJaw[0] + vctJaw[16]) / 2);
                    double angFace = atan((double)(vctJaw[8] - ptJawCenter).x /
                                          (double)(ptJawCenter - vctJaw[8]).y);
                    int jawWidth = int(norm(vctJaw[0] - vctJaw[16]));
                    int jawHeight = int(norm(ptJawCenter - vctJaw[8]));
                    double angForeheadStart = 180 - angFace * 180. / pi -
                                              atan((double)(vctJaw[0] - ptJawCenter).y /
                                                   (double)(ptJawCenter - vctJaw[0]).x) * 180. / pi;
                    double angForeheadEnd = 360 - angFace * 180. / pi -
                                            atan((double)(ptJawCenter - vctJaw[16]).y /
                                                 (double)(vctJaw[16] - ptJawCenter).x)*180. / pi;
                    ellipse2Poly(ptJawCenter, Size(jawWidth / 2, int(jawHeight / 1.5)),
                                 int(angFace * 180. / pi), int(angForeheadStart),
                                 int(angForeheadEnd), 1, vctFace);
                    size_t lenJaw = vctJaw.size();
                    for (size_t k = 0ul; k < lenJaw; k++)
                    {
                        vctFace.push_back(vctJaw[lenJaw - k - 1]);
                    }
                }

                std::vector<Point> vctLeftEye;
                {
                    Point ptLeftEyeCenter((ptsFaceElems[0] + ptsFaceElems[1]) / 2);
                    double angLeftEye = atan((double)(ptsFaceElems[0] - ptsFaceElems[1]).y /
                                        (double)(ptsFaceElems[0] - ptsFaceElems[1]).x) * 180. / pi;
                    ellipse2Poly(ptLeftEyeCenter,
                                 Size(int(norm(ptsFaceElems[0] - ptsFaceElems[1]) / 2),
                                      int(norm(ptsFaceElems[0] - ptsFaceElems[1]) / 6)),
                                 int(angLeftEye), 0, 180, 1, vctLeftEye);
                }
                vctLeftEye.push_back(ptsFaceElems[12]);
                vctLeftEye.push_back(ptsFaceElems[13]);
                vctLeftEye.push_back(ptsFaceElems[14]);

                std::vector<Point> vctRightEye;
                {
                    Point ptRightEyeCenter((ptsFaceElems[2] + ptsFaceElems[3]) / 2);
                    double angRightEye = atan((double)(ptsFaceElems[3] - ptsFaceElems[2]).y /
                                         (double)(ptsFaceElems[3] - ptsFaceElems[2]).x) * 180. / pi;
                    ellipse2Poly(ptRightEyeCenter,
                                 Size(int(norm(ptsFaceElems[3] - ptsFaceElems[2]) / 2),
                                      int(norm(ptsFaceElems[3] - ptsFaceElems[2]) / 6)),
                                 int(angRightEye), 0, 180, 1, vctRightEye);
                }
                vctRightEye.push_back(ptsFaceElems[15]);
                vctRightEye.push_back(ptsFaceElems[16]);
                vctRightEye.push_back(ptsFaceElems[17]);

                std::vector<Point> vctNose;
                vctNose.push_back(ptsFaceElems[4]);
                vctNose.push_back(ptsFaceElems[6]);
                vctNose.push_back(ptsFaceElems[5]);
                vctNose.push_back(ptsFaceElems[7]);

                std::vector<Point> vctMouth;
                {
                    std::vector<Point> vctMouthTop;
                    Point ptMouthCenter((ptsFaceElems[8] + ptsFaceElems[9]) / 2);
                    double angMouth = atan((double)(ptsFaceElems[9] - ptsFaceElems[8]).y /
                                      (double)(ptsFaceElems[9] - ptsFaceElems[8]).x) * 180. / pi;
                    ellipse2Poly(ptMouthCenter,
                                 Size(int(norm(ptsFaceElems[9] - ptsFaceElems[8]) / 2),
                                      int(norm(ptMouthCenter - ptsFaceElems[11]))),
                                 int(angMouth), 0, 180, 1, vctMouth);
                    ellipse2Poly(ptMouthCenter,
                                 Size(int(norm(ptsFaceElems[9] - ptsFaceElems[8]) / 2),
                                      int(norm(ptMouthCenter - ptsFaceElems[10]))),
                                 int(angMouth), 180, 360, 1, vctMouthTop);
                    size_t lenMouthTop = vctMouthTop.size();
                    for (size_t k = 0ul; k < lenMouthTop; k++)
                    {
                        vctMouth.push_back(vctMouthTop[k]);
                    }
                }

                //Masks
                Mat mskFace(img.rows,img.cols, CV_8UC3, Scalar(0, 0, 0));
                Mat mskBlur(img.rows,img.cols, CV_8UC3, Scalar(0, 0, 0));
                Mat mskSharp(img.rows,img.cols, CV_8UC3, Scalar(0, 0, 0));
                fillPoly(mskSharp, std::vector<std::vector<Point>>{vctLeftEye}, Scalar(255, 255, 255));
                fillPoly(mskSharp, std::vector<std::vector<Point>>{vctRightEye}, Scalar(255, 255, 255));
                fillPoly(mskSharp, std::vector<std::vector<Point>>{vctMouth}, Scalar(255, 255, 255));
                fillPoly(mskSharp, std::vector<std::vector<Point>>{vctNose}, Scalar(255, 255, 255));
                fillPoly(mskBlur, std::vector<std::vector<Point>>{vctFace}, Scalar(255, 255, 255));
                mskSharp.copyTo(mskFace);
                fillPoly(mskFace, std::vector<std::vector<Point>>{vctFace}, Scalar(255, 255, 255));
                Mat mskBlurGaussed(img.rows,img.cols, CV_8UC3, Scalar(0, 0, 0));
                Mat mskSharpGaussed(img.rows,img.cols, CV_8UC3, Scalar(0, 0, 0));
                Mat mskFaceGaussed(img.rows,img.cols, CV_8UC3, Scalar(0, 0, 0));
                GaussianBlur(mskBlur, mskBlurGaussed, Size(5, 5), 0);
                GaussianBlur(mskSharp, mskSharpGaussed, Size(5, 5), 0);
                GaussianBlur(mskFace, mskFaceGaussed, Size(5, 5), 0);
                threshold(mskBlurGaussed, mskBlurGaussed, 0, 1, THRESH_BINARY);
                threshold(mskSharpGaussed, mskSharpGaussed, 0, 1, THRESH_BINARY);
                threshold(mskFaceGaussed, mskFaceGaussed, 0, 1, THRESH_BINARY);
                mskNoFaces -= mskFaceGaussed;
                mskSharps = mskSharps - mskSharps.mul(mskSharpGaussed) + mskSharpGaussed;
                mskBlurs = mskBlurs - mskBlurs.mul(mskBlurGaussed) + mskBlurGaussed;
                mskBlurs -= mskBlurs.mul(mskSharps);

                if (flgLandmarks == true)
                {
                    std::vector<std::vector<Point>> vctvctContours;
                    vctvctContours.push_back(vctFace);
                    vctvctContours.push_back(vctLeftEye);
                    vctvctContours.push_back(vctRightEye);
                    vctvctContours.push_back(vctMouth);
                    vctvctContours.push_back(vctNose);
                    polylines(imgDraw, vctvctContours, true, clrYellow);
                }
                if (flgBoxes == true)
                {
                    rectangle(imgDraw, face,
                              clrGreen, 1);
                }
            }
            else
            {
                break;
            }
        }

        Mat imgBilat;
        Mat imgSharp;
        bilateralFilter(img, imgBilat, 9, 30, 30);
        custom::unsharpMask(img, imgSharp, 3, 1);

        Mat imgShow = img.mul(mskNoFaces) + imgBilat.mul(mskBlurs) + imgSharp.mul(mskSharps);

        imshow(winInput, imgDraw);
        imshow(winFB, imgShow);
    }
}
