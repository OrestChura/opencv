// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation


#include "precomp.hpp"

#include <opencv2/gapi/video.hpp>
#include <opencv2/gapi/cpu/video.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>

#ifdef HAVE_OPENCV_VIDEO
#include <opencv2/video.hpp>
#endif // HAVE_OPENCV_VIDEO

#ifdef HAVE_OPENCV_VIDEO

GAPI_OCV_KERNEL(GCPUBuildOptFlowPyramid, cv::gapi::video::GBuildOptFlowPyramid)
{
    static void run(const cv::Mat              &img,
                    const cv::Size             &winSize,
                    const cv::Scalar           &maxLevel,
                          bool                  withDerivatives,
                          int                   pyrBorder,
                          int                   derivBorder,
                          bool                  tryReuseInputImage,
                          std::vector<cv::Mat> &outPyr,
                          cv::Scalar           &outMaxLevel)
    {
        outMaxLevel = cv::buildOpticalFlowPyramid(img, outPyr, winSize,
                                                  static_cast<int>(maxLevel[0]),
                                                  withDerivatives, pyrBorder,
                                                  derivBorder, tryReuseInputImage);
    }
};

GAPI_OCV_KERNEL(GCPUCalcOptFlowLK, cv::gapi::video::GCalcOptFlowLK)
{
    static void run(const cv::Mat                  &prevImg,
                    const cv::Mat                  &nextImg,
                    const std::vector<cv::Point2f> &prevPts,
                    const std::vector<cv::Point2f> &predPts,
                    const cv::Size                 &winSize,
                    const cv::Scalar               &maxLevel,
                    const cv::TermCriteria         &criteria,
                          int                       flags,
                          double                    minEigThresh,
                          std::vector<cv::Point2f> &outPts,
                          std::vector<uchar>       &status,
                          std::vector<float>       &err)
    {
        if (flags & cv::OPTFLOW_USE_INITIAL_FLOW)
            outPts = predPts;
        cv::calcOpticalFlowPyrLK(prevImg, nextImg, prevPts, outPts, status, err, winSize,
                                 static_cast<int>(maxLevel[0]), criteria, flags, minEigThresh);
    }
};

GAPI_OCV_KERNEL(GCPUCalcOptFlowLKForPyr, cv::gapi::video::GCalcOptFlowLKForPyr)
{
    static void run(const std::vector<cv::Mat>     &prevPyr,
                    const std::vector<cv::Mat>     &nextPyr,
                    const std::vector<cv::Point2f> &prevPts,
                    const std::vector<cv::Point2f> &predPts,
                    const cv::Size                 &winSize,
                    const cv::Scalar               &maxLevel,
                    const cv::TermCriteria         &criteria,
                          int                       flags,
                          double                    minEigThresh,
                          std::vector<cv::Point2f> &outPts,
                          std::vector<uchar>       &status,
                          std::vector<float>       &err)
    {
        if (flags & cv::OPTFLOW_USE_INITIAL_FLOW)
            outPts = predPts;
        cv::calcOpticalFlowPyrLK(prevPyr, nextPyr, prevPts, outPts, status, err, winSize,
                                 static_cast<int>(maxLevel[0]), criteria, flags, minEigThresh);
    }
};

GAPI_OCV_KERNEL_ST(GCPUBackgroundSubtractorMOG2,
                   cv::gapi::video::GBackgroundSubtractorMOG2,
                   cv::BackgroundSubtractorMOG2)
{
    static void setup(const cv::GMatDesc &, double,
                      std::shared_ptr<cv::BackgroundSubtractorMOG2> &state,
                      const cv::GCompileArgs &compileArgs)
    {
        auto bsParams = cv::gapi::getCompileArg<cv::gapi::video::BackgroundSubtractorParams>(compileArgs)
                           .value_or(cv::gapi::video::BackgroundSubtractorParams{});

        state = cv::createBackgroundSubtractorMOG2(bsParams.history,
                                                   bsParams.threshold,
                                                   bsParams.detectShadows);

        GAPI_Assert(state);
    }

    static void run(const cv::Mat& in, double learningRate, cv::Mat &out, cv::BackgroundSubtractorMOG2& state)
    {
        state.apply(in, out, learningRate);
    }
};

GAPI_OCV_KERNEL_ST(GCPUKalmanFilter, cv::gapi::video::GKalmanFilter, cv::KalmanFilter)
{
    static void setup(const cv::GMatDesc &, const cv::GMatDesc &,
                      std::shared_ptr<cv::KalmanFilter> &state,
                      const cv::GCompileArgs &compileArgs)
    {

        auto kfParams = cv::gapi::getCompileArg<cv::gapi::video::KalmanParams>(compileArgs)
                        .value_or(cv::gapi::video::KalmanParams{});

        state = std::make_shared<cv::KalmanFilter>(kfParams.dpDims, kfParams.mpDims);

        state->transitionMatrix = kfParams.transitionMatrix;
        state->statePre = kfParams.statePre;
        state->statePost = kfParams.statePost;
        state->measurementMatrix = kfParams.measurementMatrix;
        state->processNoiseCov = kfParams.processNoiseCov;
        state->measurementNoiseCov = kfParams.measurementNoiseCov;
        state->errorCovPre = kfParams.errorCovPre;
        state->gain = kfParams.gain;
        state->errorCovPost = kfParams.errorCovPost;
        state->controlMatrix = kfParams.controlMatrix;

        GAPI_Assert(state);
    }

    static void run(const cv::Mat& measurements, const cv::Mat& control, cv::Mat &out, cv::KalmanFilter& state)
    {
        cv::Mat pre;

        if (!control.empty() && cv::countNonZero(control) > 0)
            pre = state.predict(control);
        else
            pre = state.predict();

        if (!measurements.empty() && cv::countNonZero(measurements) > 0)
            state.correct(measurements).copyTo(out);
        else
            pre.copyTo(out);
    }
};

cv::gapi::GKernelPackage cv::gapi::video::cpu::kernels()
{
    static auto pkg = cv::gapi::kernels
        < GCPUBuildOptFlowPyramid
        , GCPUCalcOptFlowLK
        , GCPUCalcOptFlowLKForPyr
        , GCPUBackgroundSubtractorMOG2
        , GCPUKalmanFilter
        >();
    return pkg;
}

#else

cv::gapi::GKernelPackage cv::gapi::video::cpu::kernels()
{
    return GKernelPackage();
}

#endif // HAVE_OPENCV_VIDEO
