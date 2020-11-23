// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#ifndef OPENCV_GAPI_VIDEO_TESTS_INL_HPP
#define OPENCV_GAPI_VIDEO_TESTS_INL_HPP

#include "gapi_video_tests.hpp"
#include <opencv2/gapi/streaming/cap.hpp>
#include "opencv2/ts.hpp"

namespace opencv_test
{

TEST_P(BuildOptFlowPyramidTest, AccuracyTest)
{
    std::vector<Mat> outPyrOCV,          outPyrGAPI;
    int              outMaxLevelOCV = 0, outMaxLevelGAPI = 0;

    BuildOpticalFlowPyramidTestParams params { fileName, winSize, maxLevel,
                                              withDerivatives, pyrBorder, derivBorder,
                                              tryReuseInputImage, getCompileArgs() };

    BuildOpticalFlowPyramidTestOutput outOCV  { outPyrOCV,  outMaxLevelOCV };
    BuildOpticalFlowPyramidTestOutput outGAPI { outPyrGAPI, outMaxLevelGAPI };

    runOCVnGAPIBuildOptFlowPyramid(*this, params, outOCV, outGAPI);

    compareOutputPyramids(outOCV, outGAPI);
}

TEST_P(OptFlowLKTest, AccuracyTest)
{
    std::vector<cv::Point2f> outPtsOCV,    outPtsGAPI,    inPts;
    std::vector<uchar>       outStatusOCV, outStatusGAPI;
    std::vector<float>       outErrOCV,    outErrGAPI;

    OptFlowLKTestParams params { fileNamePattern, channels, pointsNum,
                                 winSize, criteria, getCompileArgs() };

    OptFlowLKTestOutput outOCV  { outPtsOCV,  outStatusOCV,  outErrOCV };
    OptFlowLKTestOutput outGAPI { outPtsGAPI, outStatusGAPI, outErrGAPI };

    runOCVnGAPIOptFlowLK(*this, inPts, params, outOCV, outGAPI);

    compareOutputsOptFlow(outOCV, outGAPI);
}

TEST_P(OptFlowLKTestForPyr, AccuracyTest)
{
    std::vector<cv::Mat>     inPyr1, inPyr2;
    std::vector<cv::Point2f> outPtsOCV,    outPtsGAPI,    inPts;
    std::vector<uchar>       outStatusOCV, outStatusGAPI;
    std::vector<float>       outErrOCV,    outErrGAPI;

    OptFlowLKTestParams params { fileNamePattern, channels, pointsNum,
                                 winSize, criteria, getCompileArgs() };

    OptFlowLKTestInput<std::vector<cv::Mat>> in { inPyr1, inPyr2, inPts };
    OptFlowLKTestOutput outOCV  { outPtsOCV,  outStatusOCV,  outErrOCV };
    OptFlowLKTestOutput outGAPI { outPtsGAPI, outStatusGAPI, outErrGAPI };

    runOCVnGAPIOptFlowLKForPyr(*this, in, params, withDeriv, outOCV, outGAPI);

    compareOutputsOptFlow(outOCV, outGAPI);
}

TEST_P(BuildPyr_CalcOptFlow_PipelineTest, AccuracyTest)
{
    std::vector<Point2f> outPtsOCV,    outPtsGAPI,    inPts;
    std::vector<uchar>   outStatusOCV, outStatusGAPI;
    std::vector<float>   outErrOCV,    outErrGAPI;

    BuildOpticalFlowPyramidTestParams params { fileNamePattern, winSize, maxLevel,
                                              withDerivatives, BORDER_DEFAULT, BORDER_DEFAULT,
                                              true, getCompileArgs() };

    auto customKernel  = gapi::kernels<GCPUMinScalar>();
    auto kernels       = gapi::combine(customKernel,
                                       params.compileArgs[0].get<gapi::GKernelPackage>());
    params.compileArgs = compile_args(kernels);

    OptFlowLKTestOutput outOCV  { outPtsOCV,  outStatusOCV,  outErrOCV };
    OptFlowLKTestOutput outGAPI { outPtsGAPI, outStatusGAPI, outErrGAPI };

    runOCVnGAPIOptFlowPipeline(*this, params, outOCV, outGAPI, inPts);

    compareOutputsOptFlow(outOCV, outGAPI);
}

inline cv::GCompileArgs getCompileArgsUpdate(cv::GCompileArgs&& args, const GCompileArg& obj)
{
    args.emplace_back(obj);
    return args;
}

TEST_P(BackgroundSubtractorMOG2Test, AccuracyTest)
{
    initTestDataPath();

    // G-API graph declaration
    cv::GMat in;
    cv::GMat out = cv::gapi::BackgroundSubtractorMOG2(in, -1);
    // Preserving 'in' in output to have possibility to compare with OpenCV reference
    cv::GComputation c(cv::GIn(in), cv::GOut(cv::gapi::copy(in), out));

    // G-API compilation of graph for streaming mode
    auto gapiBackSub = c.compileStreaming(getCompileArgs());

    // Testing G-API Background Substractor in streaming mode
    try
    {
        gapiBackSub.setSource(gapi::wip::make_src<cv::gapi::wip::GCaptureSource>
                                                      (findDataFile(filePath1)));
    }
    catch (...)
    { throw SkipTestException("Video file can't be opened."); }

    // Allowing 1% difference of all pixels between G-API and reference OpenCV results
    testBackSubInStreaming(gapiBackSub, 1);
}


TEST_P(KalmanFilterTest, AccuracyTest)
{
    const int Dim = 7;
    const double max_init = 1;
    //const double max_noise = 0.1;
    const double eps = 1.000;

    cv::Mat measure(Dim, 1, CV_32F);
    cv::Mat control = Mat::zeros(Dim, Dim, CV_32F);

    cv::Mat gapiKState(Dim, 1, CV_32F);
    cv::Mat ocvKState(Dim, 1, CV_32F);

    cv::randu(measure, Scalar::all(-max_init), Scalar::all(max_init));

    // G-API code
    cv::GMat m, contr;
    cv::GMat out = cv::gapi::KalmanFilter(m, contr);
    cv::GComputation c(cv::GIn(m, contr), cv::GOut(out));

    std::vector<cv::Mat> mats = { measure, control };

    auto gapiKalman = c.compile(cv::descrs_of(mats), getCompileArgs());

    gapiKalman(cv::gin(measure, control), cv::gout(gapiKState));

    cv::KalmanFilter ocvKalman(Dim, Dim, 0, CV_32F);
    ocvKalman.predict();
    ocvKState = ocvKalman.correct(measure);

    // Comparison //////////////////////////////////////////////////////////////
    {
        double diff = 0;
        vector<int> idx;
        int code = cmpEps(gapiKState, ocvKState, &diff, eps, &idx, false);
        EXPECT_TRUE(code >= 0);
    }
}
} // opencv_test

#endif // OPENCV_GAPI_VIDEO_TESTS_INL_HPP
