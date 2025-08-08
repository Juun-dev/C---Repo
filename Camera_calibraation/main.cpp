#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

// Function to prepare object points for calibration
cv::Mat prepareObjectPoints(int rows, int cols, float squareSize) {
    cv::Mat objp(rows * cols, 3, CV_32F);
    for(int i = 0; i < rows; i++)
        for(int j = 0; j < cols; j++) {
            objp.at<float>(i*cols + j, 0) = j * squareSize;
            objp.at<float>(i*cols + j, 1) = i * squareSize;
            objp.at<float>(i*cols + j, 2) = 0.0f;
        }
    return objp;
}

// Save calibration including extrinsics and scale map
void saveCalibration(const std::string& filename,
                     const cv::Mat& cameraMatrix,
                     const cv::Mat& distCoeffs,
                     const std::vector<cv::Mat>& rvecs,
                     const std::vector<cv::Mat>& tvecs,
                     const cv::Mat& scaleMap)
{
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    fs << "camera_matrix" << cameraMatrix;
    fs << "distortion_coefficients" << distCoeffs;
    fs << "rvecs" << "[";
    for (const auto& r : rvecs) fs << r;
    fs << "]";
    fs << "tvecs" << "[";
    for (const auto& t : tvecs) fs << t;
    fs << "]";
    fs << "scale_map" << scaleMap;
    fs.release();
    std::cout << "Calibration data saved to " << filename << std::endl;
}

// Generate pixel-to-mm scale map across the image
cv::Mat generatePixelMmMap(const cv::Mat& cameraMatrix,
                           const cv::Mat& distCoeffs,
                           const cv::Size& imageSize,
                           float squareSize,
                           int gridSpacing = 20)
{
    // We'll sample points on the checkerboard plane Z=0, covering image field
    int rows = (imageSize.height / gridSpacing) + 1;
    int cols = (imageSize.width / gridSpacing) + 1;
    cv::Mat scaleMap(rows, cols, CV_32F);

    // rvec and tvec to identity (looking straight at plane at Z=1000mm)
    cv::Mat rvec = cv::Mat::zeros(3,1,CV_64F);
    cv::Mat tvec = (cv::Mat_<double>(3,1) << 0, 0, 1000);

    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            // Define two nearby world points 1 squareSize apart in X direction at Z=0 plane
            std::vector<cv::Point3f> worldPoints;
            worldPoints.push_back(cv::Point3f(j * gridSpacing * squareSize, i * gridSpacing * squareSize, 0));
            worldPoints.push_back(cv::Point3f(j * gridSpacing * squareSize + squareSize, i * gridSpacing * squareSize, 0));

            std::vector<cv::Point2f> imagePoints;
            cv::projectPoints(worldPoints, rvec, tvec, cameraMatrix, distCoeffs, imagePoints);

            double pixelDist = cv::norm(imagePoints[0] - imagePoints[1]);
            float pixelsPerMm = static_cast<float>(pixelDist / squareSize);
            scaleMap.at<float>(i, j) = pixelsPerMm;
        }
    }
    return scaleMap;
}

int main() {
    int rows, cols;
    float squareSize;
    std::string imagesPathPattern;

    std::cout << "Enter number of inner corners (rows): ";
    std::cin >> rows;
    std::cout << "Enter number of inner corners (columns): ";
    std::cin >> cols;
    std::cout << "Enter square size in mm: ";
    std::cin >> squareSize;
    std::cout << "Enter path pattern to calibration images (e.g. ./images/*.jpg): ";
    std::cin >> imagesPathPattern;

    std::vector<std::string> images;
    cv::glob(imagesPathPattern, images);

    if (images.size() < 6) {
        std::cerr << "Need at least 6 calibration images with different poses!" << std::endl;
        return -1;
    }

    cv::Mat objp = prepareObjectPoints(rows, cols, squareSize);
    std::vector<std::vector<cv::Point3f>> objpoints;
    std::vector<std::vector<cv::Point2f>> imgpoints;
    cv::Size imgSize;

    cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001);

    for (auto& file : images) {
        cv::Mat img = cv::imread(file);
        if (img.empty()) {
            std::cerr << "Failed to load image " << file << std::endl;
            continue;
        }
        if (imgSize == cv::Size()) imgSize = img.size();

        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> corners;
        bool found = cv::findChessboardCorners(gray, cv::Size(cols, rows), corners,
                                               cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE);

        if (found) {
            cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1), criteria);
            imgpoints.push_back(corners);

            std::vector<cv::Point3f> objp_vec;
            objp_vec.assign((cv::Point3f*)objp.data, (cv::Point3f*)objp.data + objp.rows);
            objpoints.push_back(objp_vec);

            cv::drawChessboardCorners(img, cv::Size(cols, rows), corners, found);
            cv::imshow("Chessboard", img);
            cv::waitKey(200);
        }
        else {
            std::cout << "Chessboard not found in " << file << std::endl;
        }
    }

    cv::destroyAllWindows();

    cv::Mat cameraMatrix, distCoeffs;
    std::vector<cv::Mat> rvecs, tvecs;
    double rms = cv::calibrateCamera(objpoints, imgpoints, imgSize, cameraMatrix,
                                     distCoeffs, rvecs, tvecs);

    std::cout << "\nCalibration done with RMS error = " << rms << std::endl;
    std::cout << "Camera matrix:\n" << cameraMatrix << std::endl;
    std::cout << "Distortion coefficients:\n" << distCoeffs.t() << std::endl;

    // Generate pixel-to-mm scale map
    cv::Mat scaleMap = generatePixelMmMap(cameraMatrix, distCoeffs, imgSize, squareSize, 20);

    // Save all calibration data including scale map
    saveCalibration("calibration_with_scale.yml", cameraMatrix, distCoeffs, rvecs, tvecs, scaleMap);

    // Optionally display scale map as image (normalized for visualization)
    cv::Mat scaleMapVis;
    cv::normalize(scaleMap, scaleMapVis, 0, 255, cv::NORM_MINMAX);
    scaleMapVis.convertTo(scaleMapVis, CV_8U);
    cv::applyColorMap(scaleMapVis, scaleMapVis, cv::COLORMAP_JET);
    cv::imshow("Pixel/mm Scale Map", scaleMapVis);
    cv::waitKey(0);

    return 0;
}
