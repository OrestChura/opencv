// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#ifndef OPENCV_GAPI_CPU_VIDEO_API_HPP
#define OPENCV_GAPI_CPU_VIDEO_API_HPP

#include <opencv2/gapi/gkernel.hpp> // GKernelPackage
#include "opencv2/core/mat.hpp"
#include "opencv2/core.hpp"

#undef countNonZero

namespace cv {
namespace gapi {
namespace video {
namespace cpu {

GAPI_EXPORTS GKernelPackage kernels();

} // namespace cpu

struct KalmanParams
{
    KalmanParams()
    {
        statePre = Mat::zeros(dpDims, 1, type);
        statePost = Mat::zeros(dpDims, 1, type);
        transitionMatrix = Mat::eye(dpDims, dpDims, type);

        processNoiseCov = Mat::eye(dpDims, dpDims, type);
        measurementMatrix = Mat::zeros(mpDims, dpDims, type);
        measurementNoiseCov = Mat::eye(mpDims, mpDims, type);

        errorCovPre = Mat::zeros(dpDims, dpDims, type);
        errorCovPost = Mat::zeros(dpDims, dpDims, type);
        gain = Mat::zeros(dpDims, mpDims, type);

        controlMatrix = Mat::zeros(dpDims, controlDims, type);
    }

    KalmanParams(int dp, int mp, int cp = 0, int depth = CV_32F)
    {
        GAPI_Assert(dp > 0 && mp > 0);
        GAPI_Assert(depth == CV_32F || depth == CV_64F);
        controlDims = std::max(cp, 0);
        type = depth;
        dpDims = dp;
        mpDims = mp;

        statePre = Mat::zeros(dp, 1, type);
        statePost = Mat::zeros(dp, 1, type);
        transitionMatrix = Mat::eye(dp, dp, type);

        processNoiseCov = Mat::eye(dp, dp, type);
        measurementMatrix = Mat::zeros(mp, dp, type);
        measurementNoiseCov = Mat::eye(mp, mp, type);

        errorCovPre = Mat::zeros(dp, dp, type);
        errorCovPost = Mat::zeros(dp, dp, type);
        gain = Mat::zeros(dp, mp, type);

        if (controlDims > 0)
            controlMatrix = Mat::zeros(dp, controlDims, type);
        else
            controlMatrix.release();
    }

    KalmanParams(Mat& sPre, Mat& sPost,
                 const Mat& transMat, Mat& prNoiseCov, Mat& measureMat, Mat& measureNoiseCov,
                 Mat& errCovPre, Mat& errCovPost, Mat& g, Mat& contrMat, int cpDim, int dpDim, int mpDim)
    {
        GAPI_Assert(dpDim > 0 && mpDim > 0);
        dpDims = dpDim;
        mpDims = mpDim;
        controlDims = std::max(cpDim, 0);

        if (!transMat.empty() && (cv::countNonZero(transMat) > 0))
        {
            GAPI_Assert(transMat.depth() == CV_32F || transMat.depth() == CV_64F);
            GAPI_Assert(transMat.rows == transMat.cols);

            transitionMatrix = transMat;
            type = transMat.depth();
        }
        else
            transitionMatrix = Mat::eye(dpDims, dpDims, type);

        if (!prNoiseCov.empty() && (countNonZero(prNoiseCov) > 0))
        {
            GAPI_Assert((prNoiseCov.depth() == type) &&
                        ((prNoiseCov.rows == prNoiseCov.cols) == dpDims));

            processNoiseCov = prNoiseCov;
        }
        else
            processNoiseCov = Mat::eye(dpDims, dpDims, type);

        if (!sPre.empty())
        {
            GAPI_Assert(sPre.depth() == type && sPre.rows == dpDims && sPre.cols == 1);
            statePre = sPre;
        }
        else
            statePre = Mat::zeros(dpDims, 1, type);

        if (!sPost.empty())
        {
            GAPI_Assert(sPost.depth() == type && sPost.rows == dpDims &&
                        sPost.cols == 1);
            statePost = sPost;
        }
        else
            statePost = Mat::zeros(dpDims, 1, type);

        if (!errCovPre.empty())
        {
            GAPI_Assert((errCovPre.depth() == type) &&
                        ((errCovPre.rows == errCovPre.cols) == dpDims));
            errorCovPre = errCovPre;
        }
        else
            errorCovPre = Mat::zeros(dpDims, dpDims, type);

        if (!errCovPost.empty())
        {
            GAPI_Assert((errCovPost.depth() == type) &&
                        ((errCovPost.rows == errCovPost.cols) == dpDims));
            errorCovPost = errCovPost;
        }
        else
            errorCovPost = Mat::zeros(dpDims, dpDims, type);

        if(!measureMat.empty())
        {
            GAPI_Assert(measureMat.depth() == type && measureMat.cols == dpDims &&
                        measureMat.rows == mpDims);
            measurementMatrix = measureMat;
        }
        else
            measurementMatrix = Mat::zeros(mpDims, dpDims, type);

        if (!g.empty())
        {
            GAPI_Assert(g.depth() == type && g.rows == dpDims && g.cols == mpDims);
            gain = g;
        }
        else
            gain = Mat::zeros(dpDims, mpDims, type);

        if (!contrMat.empty() || cpDim > 0)
        {
            GAPI_Assert(contrMat.depth() == type && contrMat.rows == dpDims && contrMat.rows == controlDims);
            controlMatrix = contrMat;
        }
        else
            controlMatrix.release();

        if (!measureNoiseCov.empty() && (countNonZero(measureNoiseCov) > 0))
        {
            GAPI_Assert((measureNoiseCov.depth() == type) &&
                        ((measureNoiseCov.rows == measureNoiseCov.cols) == mpDims));
            measurementNoiseCov = measureNoiseCov;
        }
        else
            measurementNoiseCov = Mat::eye(mpDims, mpDims, type);
    }

    int type = CV_32F;
    int controlDims = 1;
    int dpDims = 1;
    int mpDims = 1;
    Mat statePre;
    Mat statePost;
    Mat transitionMatrix;
    Mat controlMatrix;
    Mat measurementMatrix;
    Mat processNoiseCov;
    Mat measurementNoiseCov;
    Mat errorCovPre;
    Mat gain;
    Mat errorCovPost;
};

struct BackgroundSubtractorParams
{
    int history;
    double threshold;
    bool detectShadows;

    BackgroundSubtractorParams()
    {
        history = 500;
        threshold = 16;
        detectShadows = true;
    }

    BackgroundSubtractorParams(int hist, double thr, bool detect)
    {
        history = hist;
        threshold = thr;
        detectShadows = detect;
    }
};
} // namespace video
} // namespace gapi

namespace detail {
    template<> struct CompileArgTag<cv::gapi::video::BackgroundSubtractorParams>
    {
        static const char* tag()
        {
            return "org.opencv.video.background_substractor_params";
        }
    };

    template<> struct CompileArgTag<cv::gapi::video::KalmanParams>
    {
        static const char* tag()
        {
            return "org.opencv.video.kalman_params";
        }
    };
}  // namespace detail
} // namespace cv

#endif // OPENCV_GAPI_CPU_VIDEO_API_HPP
