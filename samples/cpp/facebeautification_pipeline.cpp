//  This sample is an OpenCV implementation of a face beautifiaction algorythm.
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

void unsharpMask(const Mat& src, Mat& dst, int sigma, double strength)
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

int main(int argc, char** argv)
{
    const std::string winName = "FaceBeautificator";
    namedWindow(winName, WINDOW_NORMAL);
    namedWindow("Input", WINDOW_NORMAL);
    const Scalar clrGreen(0, 255, 0);
    const Scalar clrYellow(0, 255, 255);
    const Scalar clrBlack(0, 0, 0);
    const double pi = 3.1415926535897;

    CommandLineParser parser(argc, argv,
     "{ help  h              | |     Print the help message. }"
     "{ facestruct           | |     Full path to a Face detection model structure file (for example, .xml file).}"
     "{ faceweights          | |     Full path to a Face detection model weights file (for example, .bin file).}"
     "{ landmstruct          | |     Full path to a facial Landmarks detection model structure file (for example, .xml file).}"
     "{ landmweights         | |     Full path to a facial Landmarks detection model weights file (for example, .bin file).}"
     "{ input          i     | |     Full path to an input image or a video file. Skip this argument to capture frames from a camera.}"
     );

    parser.about("Use this script to run face beautification algorythm.");
    if (argc == 1 || parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    //Parsing input arguments
    const std::string faceXmlPath = parser.get<std::string>("facestruct");
    const std::string faceBinPath = parser.get<std::string>("faceweights");

    const std::string landmXmlPath = parser.get<std::string>("landmstruct");
    const std::string landmBinPath = parser.get<std::string>("landmweights");

    //Models' definition & initialization
    Net faceNet = readNet(faceXmlPath, faceBinPath);
    const unsigned int faceObjectSize = 7;
    const float faceConfThreshold = 0.7f;
    const unsigned int faceCols = 672;
    const unsigned int faceRows = 384;

    Net landmNet = readNet(landmXmlPath, landmBinPath);
    const unsigned int landmCols = 60;
    const unsigned int landmRows = 60;

    //Input
    VideoCapture cap;
    if (parser.has("input"))
        cap.open(parser.get<String>("input"));
    else if (!cap.open(0))
            {
                std::cout << "No input available" << std::endl;
                return 1;
            }

    Mat img;
    while (waitKey(1) < 0)
    {
        cap >> img;
        if (img.empty())
        {
           waitKey();
           break;
        }

        //Infering Face detector
        faceNet.setInput(blobFromImage(img, 1.0, Size(faceCols, faceRows)));
        Mat faceOut = faceNet.forward();

        Mat mskFaces(img.rows,img.cols, CV_8UC3, Scalar(0, 0, 0));
        Mat mskBlurs(img.rows,img.cols, CV_8UC3, Scalar(0, 0, 0));
        Mat mskSharps(img.rows,img.cols, CV_8UC3, Scalar(0, 0, 0));

            //Face boxes processing
        float* faceData = (float*)(faceOut.data);
        for (size_t i = 0ul; i < faceOut.total(); i += faceObjectSize)
        {
            float faceConfidence = faceData[i + 2];
            if (faceConfidence > faceConfThreshold)
            {
                int faceLeft = int(faceData[i + 3] * img.cols);
                faceLeft = std::max(faceLeft, 0);
                int faceTop = int(faceData[i + 4] * img.rows);
                faceTop = std::max(faceTop, 0);
                int faceRight  = int(faceData[i + 5] * img.cols);
                faceRight = std::min(faceRight, img.cols - 2);
                int faceBot = int(faceData[i + 6] * img.rows);
                faceBot = std::min(faceBot, img.rows - 2);
                int faceWidth  = faceRight - faceLeft + 1;
                int faceHeight = faceBot - faceTop + 1;

                //Postprocessing for landmarks
                int faceMaxSize = std::max(faceWidth, faceHeight);
                int faceWidthAdd = faceMaxSize - faceWidth;
                int faceHeightAdd = faceMaxSize - faceHeight;

                Mat imgCrop;
                cv::copyMakeBorder(img(Rect(faceLeft, faceTop, faceWidth, faceHeight)), imgCrop, faceHeightAdd / 2, (faceHeightAdd + 1) / 2,
                                   faceWidthAdd / 2, (faceWidthAdd + 1) / 2, BORDER_CONSTANT | BORDER_ISOLATED , clrBlack);

                //Infering Landmarks detector
                landmNet.setInput(blobFromImage(imgCrop, 1.0, Size(landmCols, landmRows)));
                Mat landmOut = landmNet.forward();

                //Landmarks processing
                float* landmData = (float*)(landmOut.data);
                Point ptsFaceElems[18];
                size_t j = 0ul;
                for (; j < 18 * 2; j += 2)
                {
                    ptsFaceElems[j / 2] = Point(int(landmData[j] * imgCrop.cols + faceLeft - faceWidthAdd / 2),
                                                int(landmData[j+1] * imgCrop.rows + faceTop - faceHeightAdd / 2));
                }

                std::vector<Point> vctFace;
                {
                    std::vector<Point> vctJaw;
                    for(; j < landmOut.total(); j += 2)
                    {
                        vctJaw.push_back(Point(int(landmData[j] * imgCrop.cols + faceLeft - faceWidthAdd / 2),
                                               int(landmData[j + 1] * imgCrop.rows + faceTop - faceHeightAdd / 2)));
                    }
                    Point ptJawCenter((vctJaw[0] + vctJaw[16]) / 2);
                    double angFace = atan((double)(vctJaw[8] - ptJawCenter).x / (double)(ptJawCenter - vctJaw[8]).y);
                    int jawWidth = int(norm(vctJaw[0] - vctJaw[16]));
                    int jawHeight = int(norm(ptJawCenter - vctJaw[8]));
                    double angForeheadStart = 180 - angFace * 180. / pi -
                                              atan((double)(vctJaw[0] - ptJawCenter).y / (double)(ptJawCenter - vctJaw[0]).x) * 180. / pi;
                    double angForeheadEnd = 360 - angFace * 180. / pi -
                                            atan((double)(ptJawCenter - vctJaw[16]).y / (double)(vctJaw[16] - ptJawCenter).x)*180. / pi;
                    ellipse2Poly(ptJawCenter, Size(jawWidth / 2, int(jawHeight / 1.5)), int(angFace * 180. / pi),
                            int(angForeheadStart), int(angForeheadEnd), 1, vctFace);
                    size_t lenJaw = vctJaw.size();
                    for (size_t k = 0ul; k < lenJaw; k++)
                    {
                        vctFace.push_back(vctJaw[lenJaw - k - 1]);
                    }
                }

                std::vector<Point> vctLeftEye;
                {
                    Point ptLeftEyeCenter((ptsFaceElems[0] + ptsFaceElems[1]) / 2);
                    double angLeftEye = atan((double)(ptsFaceElems[0] - ptsFaceElems[1]).y / (double)(ptsFaceElems[0] - ptsFaceElems[1]).x) * 180. / pi;
                    ellipse2Poly(ptLeftEyeCenter, Size(int(norm(ptsFaceElems[0] - ptsFaceElems[1]) / 2), int(norm(ptsFaceElems[0] - ptsFaceElems[1]) / 6)),
                                 int(angLeftEye), 0, 180, 1, vctLeftEye);
                }
                vctLeftEye.push_back(ptsFaceElems[12]);
                vctLeftEye.push_back(ptsFaceElems[13]);
                vctLeftEye.push_back(ptsFaceElems[14]);

                std::vector<Point> vctRightEye;
                {
                    Point ptRightEyeCenter((ptsFaceElems[2] + ptsFaceElems[3]) / 2);
                    double angRightEye = atan((double)(ptsFaceElems[3] - ptsFaceElems[2]).y / (double)(ptsFaceElems[3] - ptsFaceElems[2]).x) * 180. / pi;
                    ellipse2Poly(ptRightEyeCenter, Size(int(norm(ptsFaceElems[3] - ptsFaceElems[2]) / 2), int(norm(ptsFaceElems[3] - ptsFaceElems[2]) / 6)),
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
                    double angMouth = atan((double)(ptsFaceElems[9] - ptsFaceElems[8]).y / (double)(ptsFaceElems[9] - ptsFaceElems[8]).x) * 180. / pi;
                    ellipse2Poly(ptMouthCenter, Size(int(norm(ptsFaceElems[9] - ptsFaceElems[8]) / 2), int(norm(ptMouthCenter - ptsFaceElems[11]))),
                                 int(angMouth), 0, 180, 1, vctMouth);
                    ellipse2Poly(ptMouthCenter, Size(int(norm(ptsFaceElems[9] - ptsFaceElems[8]) / 2), int(norm(ptMouthCenter - ptsFaceElems[10]))),
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
                mskBlurGaussed -= mskBlurGaussed.mul(mskSharpGaussed);
                threshold(mskBlurGaussed, mskBlurGaussed, 0, 1, THRESH_BINARY);
                threshold(mskSharpGaussed, mskSharpGaussed, 0, 1, THRESH_BINARY);
                threshold(mskFaceGaussed, mskFaceGaussed, 0, 1, THRESH_BINARY);
                mskFaces = mskFaces - mskFaces.mul(mskFaceGaussed) + mskFaceGaussed;
                mskBlurs = mskBlurs - mskBlurs.mul(mskBlurGaussed) + mskBlurGaussed;
                mskSharps = mskSharps - mskSharps.mul(mskSharpGaussed) + mskSharpGaussed;

//Uncomment the following section to draw face box and facial landmarks on the input image
//in a separate window

//                Mat imgDraw;
//                img.copyTo(imgDraw);
//                {
//                    std::vector<std::vector<Point>> vctvctContours;
//                    vctvctContours.push_back(vctFace);
//                    vctvctContours.push_back(vctLeftEye);
//                    vctvctContours.push_back(vctRightEye);
//                    vctvctContours.push_back(vctMouth);
//                    vctvctContours.push_back(vctNose);
//                    polylines(imgDraw, vctvctContours, true, clrYellow);
//                }
//                rectangle(imgDraw, Rect(faceLeft, faceTop, faceWidth, faceHeight), clrGreen, 1);
//                namedWindow("Box and landmarks", WINDOW_NORMAL);
//                imshow("Box and landmarks", imgDraw);
            }
            else
            {
                break;
            }
        }

        Mat imgBilat;
        Mat imgSharp;
        bilateralFilter(img, imgBilat, 9, 30, 30);
        unsharpMask(img, imgSharp, 3, 1);

        Mat mskNoFaces(img.rows,img.cols, CV_8UC3, Scalar(1, 1, 1));
        mskNoFaces -= mskFaces;

        Mat imgShow = img.mul(mskNoFaces) + imgBilat.mul(mskBlurs) + imgSharp.mul(mskSharps);

        imshow("Input", img);
        imshow(winName, imgShow);
    }
}
