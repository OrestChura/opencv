// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#ifndef OPENCV_GAPI_VIDEO_TESTS_COMMON_HPP
#define OPENCV_GAPI_VIDEO_TESTS_COMMON_HPP

#include "gapi_tests_common.hpp"
#include "../../include/opencv2/gapi/imgproc.hpp"

namespace
{
using Point2fVector = std::vector<cv::Point2f>;
using UcharVector   = std::vector<uchar>;
using FloatVector   = std::vector<float>;
using MatVector     = std::vector<cv::Mat>;

using GPoint2fArray = cv::GArray<cv::Point2f>;
using GUcharArray   = cv::GArray<uchar>;
using GFloatArray   = cv::GArray<float>;
using GGMatArray    = cv::GArray<cv::GMat>;
}

namespace opencv_test
{

namespace
{
void initTrackingPointsArray(Point2fVector& points, size_t width, size_t height,
                             size_t pointsXnum, size_t pointsYnum)
{
    if (pointsXnum > width || pointsYnum > height)
    {
        ADD_FAILURE() << "Specified points number is too big";
        pointsXnum = std::min(pointsXnum, width);
        pointsYnum = std::min(pointsYnum, height);
    }

    size_t stepX = width / pointsXnum;
    size_t stepY = height / pointsYnum;

    points.clear();
    points.reserve(pointsXnum * pointsYnum);

    for (size_t x = stepX / 2; x < width; x += stepX)
    {
        for (size_t y = stepY / 2; y < height; y += stepY)
        {
            Point2f pt(static_cast<float>(x), static_cast<float>(y));
            points.push_back(pt);
        }
    }
}

template<typename Type, typename GType>
std::tuple<Type,Type,Point2fVector,Point2fVector,UcharVector,FloatVector,cv::GComputation,
           Point2fVector,UcharVector,FloatVector> calcOptFlowLK_OCVnGAPI(
        Type in1, Type in2, size_t width, size_t height, std::tuple<size_t,size_t> pointsNum,
        int winSize, cv::GCompileArgs compileArgs, int maxLevel, const cv::TermCriteria& criteria,
        int flags, double minEigThreshold)
{
    Point2fVector outPtsOCV,    outPtsGAPI,    inPts;
    UcharVector   outStatusOCV, outStatusGAPI;
    FloatVector   outErrOCV,    outErrGAPI;

    initTrackingPointsArray(inPts, width, height, std::get<0>(pointsNum), std::get<1>(pointsNum));

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::calcOpticalFlowPyrLK(in1, in2, inPts, outPtsOCV, outStatusOCV, outErrOCV,
                                 cv::Size(winSize, winSize), maxLevel, criteria, flags,
                                 minEigThreshold);
    }

    // G-API code //////////////////////////////////////////////////////////////
    GType GinPrev, GinNext;
    GPoint2fArray GinPts, GpredPts, GoutPts;
    GUcharArray   Gstatus;
    GFloatArray   Gerr;
    std::tie(GoutPts, Gstatus, Gerr) = cv::gapi::calcOpticalFlowPyrLK(GinPrev, GinNext, GinPts,
                                                                      GpredPts,
                                                                      cv::Size(winSize, winSize),
                                                                      maxLevel, criteria, flags,
                                                                      minEigThreshold);
    cv::GComputation c = cv::GComputation(cv::GIn(GinPrev, GinNext, GinPts, GpredPts),
                                          cv::GOut(GoutPts, Gstatus, Gerr));

    // Warm-up graph engine:
    c.apply(cv::gin(in1, in2, inPts, Point2fVector()),
            cv::gout(outPtsGAPI, outStatusGAPI, outErrGAPI),
            std::move(compileArgs));
    return std::make_tuple(in1, in2, inPts, outPtsOCV, outStatusOCV, outErrOCV, c, outPtsGAPI,
                           outStatusGAPI, outErrGAPI);
}

inline std::tuple<cv::Mat,cv::Mat,Point2fVector,Point2fVector,UcharVector,FloatVector,
                  cv::GComputation,Point2fVector,UcharVector,FloatVector> testOptFlowLK(
        TestFunctional* const testInst, std::string fileNamePattern, int format, int channels,
        std::tuple<size_t,size_t> pointsNum, int winSize, cv::GCompileArgs compile_args,
        int maxLevel, const cv::TermCriteria& criteria, int flags, double minEigThreshold)
{
    testInst->initMatsFromImages(channels, fileNamePattern, format);
    return calcOptFlowLK_OCVnGAPI<cv::Mat, cv::GMat>(testInst->in_mat1, testInst->in_mat2,
                                                     static_cast<size_t>(testInst->in_mat1.cols),
                                                     static_cast<size_t>(testInst->in_mat1.rows),
                                                     pointsNum, winSize, compile_args, maxLevel,
                                                     criteria, flags, minEigThreshold);
}

inline std::tuple<MatVector,MatVector,Point2fVector,Point2fVector,UcharVector,FloatVector,
                  cv::GComputation,Point2fVector,UcharVector,FloatVector>testOptFlowLKForPyr(
        TestFunctional* const testInst, std::string fileNamePattern, int format, int channels,
        std::tuple<size_t,size_t> pointsNum, int winSize, bool withDeriv,
        cv::GCompileArgs compile_args, int maxLevel, const cv::TermCriteria& criteria, int flags,
        double minEigThreshold)
{
    testInst->initMatsFromImages(channels, fileNamePattern, format);

    MatVector in_pyr1, in_pyr2;
    maxLevel = cv::buildOpticalFlowPyramid(testInst->in_mat1, in_pyr1, cv::Size(winSize, winSize),
                                           maxLevel, withDeriv);
    maxLevel = cv::buildOpticalFlowPyramid(testInst->in_mat2, in_pyr2, cv::Size(winSize, winSize),
                                           maxLevel, withDeriv);
    return calcOptFlowLK_OCVnGAPI<MatVector,GGMatArray>(in_pyr1, in_pyr2,
                                                        static_cast<size_t>(testInst->in_mat1.cols),
                                                        static_cast<size_t>(testInst->in_mat1.rows),
                                                        pointsNum, winSize, compile_args, maxLevel,
                                                        criteria, flags, minEigThreshold);
}
} // namespace
} // namespace opencv_test

#endif // OPENCV_GAPI_VIDEO_TESTS_COMMON_HPP
