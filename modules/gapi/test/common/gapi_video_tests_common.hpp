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

namespace cgv = cv::gapi::video;
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

struct OptFlowLKTestParams
{
    const std::string& fileNamePattern;
    int format;
    int channels;
    const std::tuple<size_t,size_t>& pointsNum;
    int winSize;
    int maxLevel;
    const cv::TermCriteria& criteria;
    int flags;
    double minEigThreshold;
    const cv::GCompileArgs& compileArgs;
};

template<typename GType, typename Type>
cv::GComputation runOCVnGAPIOptFlowLK(size_t width, size_t height,
                                      const OptFlowLKTestParams& testParams,
                                      std::tuple<Type&,Type&,Point2fVector&>& inTuple,
                                      std::tuple<Point2fVector&,
                                                 UcharVector&,
                                                 FloatVector&>& outOCVTuple,
                                      std::tuple<Point2fVector&,
                                                 UcharVector&,
                                                 FloatVector&>& outGAPITuple)
{
    initTrackingPointsArray(std::get<2>(inTuple), width, height,
                            std::get<0>(testParams.pointsNum),
                            std::get<1>(testParams.pointsNum));

    cv::Size winSize(testParams.winSize, testParams.winSize);

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::calcOpticalFlowPyrLK(std::get<0>(inTuple), std::get<1>(inTuple), std::get<2>(inTuple),
                                 std::get<0>(outOCVTuple),
                                 std::get<1>(outOCVTuple),
                                 std::get<2>(outOCVTuple),
                                 winSize, testParams.maxLevel, testParams.criteria,
                                 testParams.flags, testParams.minEigThreshold);
    }

    // G-API code //////////////////////////////////////////////////////////////
    { 
        GType              inPrev, inNext;
        cgv::GPoint2fArray inPts,  predPts, outPts;
        cgv::GUcharArray   status;
        cgv::GFloatArray   error;
        std::tie(outPts, status, error) = cv::gapi::calcOpticalFlowPyrLK(inPrev, inNext,
                                                                         inPts, predPts,
                                                                         winSize,
                                                                         testParams.maxLevel,
                                                                         testParams.criteria,
                                                                         testParams.flags,
                                                                         testParams.minEigThreshold);

        cv::GComputation c(cv::GIn(inPrev, inNext, inPts, predPts),
                           cv::GOut(outPts, status, error));

        c.apply(cv::gin(std::get<0>(inTuple), std::get<1>(inTuple), std::get<2>(inTuple),
                        Point2fVector()),
                cv::gout(std::get<0>(outGAPITuple), std::get<1>(outGAPITuple),
                         std::get<2>(outGAPITuple)),
                std::move(const_cast<cv::GCompileArgs&>(testParams.compileArgs)));

        return c;
    }
}

inline cv::GComputation runOCVnGAPIOptFlowLK(TestFunctional& testInst,
                                             const OptFlowLKTestParams& testParams, 
                                             Point2fVector& inPts,
                                             std::tuple<Point2fVector&,
                                                        UcharVector&,
                                                        FloatVector&>& outOCVTuple,
                                             std::tuple<Point2fVector&,
                                                        UcharVector&,
                                                        FloatVector&>& outGAPITuple)
{
    testInst.initMatsFromImage(testParams.channels,
                               testParams.fileNamePattern,
                               testParams.format);

    auto inTuple = std::make_tuple(std::ref(testInst.in_mat1), std::ref(testInst.in_mat2),
                                   std::ref(inPts));
    return runOCVnGAPIOptFlowLK<cv::GMat>(static_cast<size_t>(testInst.in_mat1.cols),
                                          static_cast<size_t>(testInst.in_mat1.rows),
                                          testParams,
                                          inTuple, outOCVTuple, outGAPITuple);
}

inline cv::GComputation runOCVnGAPIOptFlowLKForPyr(TestFunctional& testInst,
                                                   const OptFlowLKTestParams& testParams,
                                                   bool withDeriv,
                                                   std::tuple<MatVector&,
                                                              MatVector&,
                                                              Point2fVector&>& inTuple,
                                                   std::tuple<Point2fVector&,
                                                              UcharVector&,
                                                              FloatVector&>& outOCVTuple,
                                                   std::tuple<Point2fVector&,
                                                              UcharVector&,
                                                              FloatVector&>& outGAPITuple)
{        
    testInst.initMatsFromImage(testParams.channels,
                               testParams.fileNamePattern,
                               testParams.format);

    cv::Size winSize(testParams.winSize, testParams.winSize);

    int maxLevel;
    maxLevel = cv::buildOpticalFlowPyramid(testInst.in_mat1, std::get<0>(inTuple),
                                           winSize, testParams.maxLevel, withDeriv);
    maxLevel = cv::buildOpticalFlowPyramid(testInst.in_mat2, std::get<1>(inTuple),
                                           winSize, testParams.maxLevel, withDeriv);

    OptFlowLKTestParams updatedTestParams(testParams);
    updatedTestParams.maxLevel = maxLevel;

    return runOCVnGAPIOptFlowLK<cgv::GGMatArray>(static_cast<size_t>(testInst.in_mat1.cols),
                                                 static_cast<size_t>(testInst.in_mat1.rows),
                                                 updatedTestParams,
                                                 inTuple, outOCVTuple, outGAPITuple);
}
} // namespace
} // namespace opencv_test

#endif // OPENCV_GAPI_VIDEO_TESTS_COMMON_HPP
