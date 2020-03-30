// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#ifndef OPENCV_GAPI_VIDEO_TESTS_COMMON_HPP
#define OPENCV_GAPI_VIDEO_TESTS_COMMON_HPP

#include "gapi_tests_common.hpp"
#include "../../include/opencv2/gapi/imgproc.hpp"


namespace opencv_test
{
namespace
{
void initTrackingPointsArray(std::vector<cv::Point2f>& points, size_t width, size_t height,
                             size_t nPointsX, size_t nPointsY)
{
    if (nPointsX > width || nPointsY > height)
    {
        FAIL() << "Specified points number is too big";
    }

    size_t stepX = width / nPointsX;
    size_t stepY = height / nPointsY;


    points.clear();
    points.reserve(nPointsX * nPointsY);

    for (size_t x = stepX / 2; x < width; x += stepX)
    {
        for (size_t y = stepY / 2; y < height; y += stepY)
        {
            Point2f pt(static_cast<float>(x), static_cast<float>(y));
            points.push_back(pt);
        }
    }
}

template<typename Type>
struct OptFlowLKTestInput
{
    Type& prevData;
    Type& nextData;
    std::vector<cv::Point2f>& prevPoints;
};

struct OptFlowLKTestOutput
{
    std::vector<cv::Point2f>& nextPoints;
    std::vector<uchar>& statuses;
    std::vector<float>& errors;
};

struct OptFlowLKTestParams
{
    OptFlowLKTestParams(): format(1), maxLevel(5), flags(0), minEigThreshold(1e-4) { }

    OptFlowLKTestParams(const std::string& namePat, int fmt, int chans,
                        const std::tuple<size_t, size_t>& ptsNum,
                        int winSz, int maxLvl, const cv::TermCriteria& crit,
                        int flgs, double minEigThresh, const cv::GCompileArgs& compArgs):
                        fileNamePattern(namePat), format(fmt), channels(chans),
                        pointsNum(ptsNum), winSize(winSz), maxLevel(maxLvl),
                        criteria(crit), flags(flgs), minEigThreshold(minEigThresh),
                        compileArgs(compArgs) { }

    std::string fileNamePattern;
    int format;
    int channels;
    std::tuple<size_t,size_t> pointsNum;
    int winSize;
    int maxLevel;
    cv::TermCriteria criteria;
    int flags;
    double minEigThreshold;
    cv::GCompileArgs compileArgs;
};

template<typename GType, typename Type>
cv::GComputation runOCVnGAPIOptFlowLK(OptFlowLKTestInput<Type>& in,
                                      size_t width, size_t height, 
                                      const OptFlowLKTestParams& params,
                                      OptFlowLKTestOutput& ocvOut,
                                      OptFlowLKTestOutput& gapiOut)
{

    int nPointsX, nPointsY;
    std::tie(nPointsX, nPointsY) = params.pointsNum;

    initTrackingPointsArray(in.prevPoints, width, height, nPointsX, nPointsY);

    cv::Size winSize(params.winSize, params.winSize);

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::calcOpticalFlowPyrLK(in.prevData, in.nextData, in.prevPoints,
                                 ocvOut.nextPoints, ocvOut.statuses, ocvOut.errors,
                                 winSize, params.maxLevel, params.criteria,
                                 params.flags, params.minEigThreshold);
    }

    // G-API code //////////////////////////////////////////////////////////////
    {
        GType                          inPrev,  inNext;
        cv::gapi::video::GPoint2fArray prevPts, predPts, nextPts;
        cv::gapi::video::GUcharArray   statuses;
        cv::gapi::video::GFloatArray   errors;
        std::tie(nextPts, statuses, errors) = cv::gapi::calcOpticalFlowPyrLK(
                                                    inPrev, inNext,
                                                    prevPts, predPts, winSize,
                                                    params.maxLevel, params.criteria,
                                                    params.flags, params.minEigThreshold);

        cv::GComputation c(cv::GIn(inPrev, inNext, prevPts, predPts),
                           cv::GOut(nextPts, statuses, errors));

        c.apply(cv::gin(in.prevData, in.nextData, in.prevPoints, std::vector<cv::Point2f>{ }),
                cv::gout(gapiOut.nextPoints, gapiOut.statuses, gapiOut.errors),
                std::move(const_cast<cv::GCompileArgs&>(params.compileArgs)));

        return c;
    }
}

inline cv::GComputation runOCVnGAPIOptFlowLK(TestFunctional& testInst,
                                             std::vector<cv::Point2f>& inPts,
                                             const OptFlowLKTestParams& params,
                                             OptFlowLKTestOutput& ocvOut,
                                             OptFlowLKTestOutput& gapiOut)
{
    testInst.initMatsFromImages(params.channels,
                                params.fileNamePattern,
                                params.format);

    OptFlowLKTestInput<cv::Mat> input{ testInst.in_mat1, testInst.in_mat2, inPts };

    return runOCVnGAPIOptFlowLK<cv::GMat>(input,
                                          static_cast<size_t>(testInst.in_mat1.cols),
                                          static_cast<size_t>(testInst.in_mat1.rows),
                                          params, ocvOut, gapiOut);
}

inline cv::GComputation runOCVnGAPIOptFlowLKForPyr(TestFunctional& testInst,
                                                   OptFlowLKTestInput<std::vector<cv::Mat>>& in,
                                                   const OptFlowLKTestParams& params,
                                                   bool withDeriv,
                                                   OptFlowLKTestOutput& ocvOut,
                                                   OptFlowLKTestOutput& gapiOut)
{
    testInst.initMatsFromImages(params.channels,
                                params.fileNamePattern,
                                params.format);

    cv::Size winSize(params.winSize, params.winSize);

    int maxLevel;
    maxLevel = cv::buildOpticalFlowPyramid(testInst.in_mat1, in.prevData,
                                           winSize, params.maxLevel, withDeriv);
    maxLevel = cv::buildOpticalFlowPyramid(testInst.in_mat2, in.nextData,
                                           winSize, params.maxLevel, withDeriv);

    OptFlowLKTestParams updatedParams(params);
    updatedParams.maxLevel = maxLevel;

    return runOCVnGAPIOptFlowLK<cv::gapi::video::GGMatArray>(
                                    in,
                                    static_cast<size_t>(testInst.in_mat1.cols),
                                    static_cast<size_t>(testInst.in_mat1.rows),
                                    updatedParams,
                                    ocvOut, gapiOut);
}
} // namespace
} // namespace opencv_test

#endif // OPENCV_GAPI_VIDEO_TESTS_COMMON_HPP
