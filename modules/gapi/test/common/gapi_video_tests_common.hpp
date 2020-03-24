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

using namespace cv::gapi::video;
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
cv::GComputation calcOptFlowLK_OCVnGAPI(size_t width, size_t height,
                                        std::tuple<size_t,size_t> pointsNum, int winSize,
                                        cv::GCompileArgs compileArgs, int maxLevel,
                                        const cv::TermCriteria& criteria, int flags,
                                        double minEigThreshold,
                                        std::tuple<Type,Type,Point2fVector>& inTuple,
                                        std::tuple<Point2fVector,
                                                   UcharVector,
                                                   FloatVector>& outOCVTuple,
                                        std::tuple<Point2fVector,
                                                   UcharVector,
                                                   FloatVector>& outGAPITuple)
{
    initTrackingPointsArray(std::get<2>(inTuple), width, height, std::get<0>(pointsNum),
                            std::get<1>(pointsNum));

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::calcOpticalFlowPyrLK(std::get<0>(inTuple), std::get<1>(inTuple), std::get<2>(inTuple),
                                 std::get<0>(outOCVTuple), std::get<1>(outOCVTuple),
                                 std::get<2>(outOCVTuple), cv::Size(winSize, winSize), maxLevel,
                                 criteria, flags, minEigThreshold);
    }

    // G-API code //////////////////////////////////////////////////////////////
    GType         GinPrev, GinNext;
    GPoint2fArray GinPts,  GpredPts, GoutPts;
    GUcharArray   Gstatus;
    GFloatArray   Gerr;
    std::tie(GoutPts, Gstatus, Gerr) = cv::gapi::calcOpticalFlowPyrLK(GinPrev, GinNext, GinPts,
                                                                      GpredPts,
                                                                      cv::Size(winSize, winSize),
                                                                      maxLevel, criteria, flags,
                                                                      minEigThreshold);
    cv::GComputation c(cv::GIn(GinPrev, GinNext, GinPts, GpredPts),
                       cv::GOut(GoutPts, Gstatus, Gerr));

    // Warm-up graph engine:
    c.apply(cv::gin(std::get<0>(inTuple), std::get<1>(inTuple), std::get<2>(inTuple),
                    Point2fVector()),
            cv::gout(std::get<0>(outGAPITuple), std::get<1>(outGAPITuple),
                     std::get<2>(outGAPITuple)),
            std::move(compileArgs));
}

inline cv::GComputation callOptFlowLK(TestFunctional* const testInst, std::string fileNamePattern,
                                      int format, int channels, std::tuple<size_t,size_t> pointsNum,
                                      int winSize, cv::GCompileArgs compileArgs, int maxLevel,
                                      const cv::TermCriteria& criteria, int flags,
                                      double minEigThreshold,
                                      Point2fVector& inPts,
                                      std::tuple<Point2fVector,
                                                 UcharVector,
                                                 FloatVector>& outOCVTuple,
                                      std::tuple<Point2fVector,
                                                 UcharVector,
                                                 FloatVector>& outGAPITuple)
{
    testInst->initMatsFromImages(channels, fileNamePattern, format);
    auto inTuple = std::make_tuple(testInst->in_mat1, testInst->in_mat2, inPts);
    return calcOptFlowLK_OCVnGAPI<cv::Mat, cv::GMat>(static_cast<size_t>(testInst->in_mat1.cols),
                                                     static_cast<size_t>(testInst->in_mat1.rows),
                                                     pointsNum, winSize, compileArgs, maxLevel,
                                                     criteria, flags, minEigThreshold,
                                                     inTuple, outOCVTuple, outGAPITuple);
}

inline cv::GComputation callOptFlowLKForPyr(TestFunctional* const testInst,
                                            std::string fileNamePattern, int format, int channels,
                                            std::tuple<size_t,size_t> pointsNum, int winSize,
                                            bool withDeriv, cv::GCompileArgs compileArgs,
                                            int maxLevel, const cv::TermCriteria& criteria,
                                            int flags, double minEigThreshold,
                                            std::tuple<MatVector,MatVector,Point2fVector>& inTuple,
                                            std::tuple<Point2fVector,
                                                       UcharVector,
                                                       FloatVector>& outOCVTuple,
                                            std::tuple<Point2fVector,
                                                       UcharVector,
                                                       FloatVector>& outGAPITuple)
{
    testInst->initMatsFromImages(channels, fileNamePattern, format);

    maxLevel = cv::buildOpticalFlowPyramid(testInst->in_mat1, std::get<0>(inTuple),
                                           cv::Size(winSize, winSize), maxLevel, withDeriv);
    maxLevel = cv::buildOpticalFlowPyramid(testInst->in_mat2, std::get<1>(inTuple),
                                           cv::Size(winSize, winSize), maxLevel, withDeriv);
    return calcOptFlowLK_OCVnGAPI<MatVector,GGMatArray>(static_cast<size_t>(testInst->in_mat1.cols),
                                                        static_cast<size_t>(testInst->in_mat1.rows),
                                                        pointsNum, winSize, compileArgs, maxLevel,
                                                        criteria, flags, minEigThreshold,
                                                        inTuple, outOCVTuple, outGAPITuple);
}
} // namespace
} // namespace opencv_test

#endif // OPENCV_GAPI_VIDEO_TESTS_COMMON_HPP
