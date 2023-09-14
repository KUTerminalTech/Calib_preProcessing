#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/calib3d/calib3d_c.h>

#include <chrono>
#include <thread>
#include <iostream>
#include <vector>
#include <unistd.h>

// #ifdef __APPLE__
#include <fstream> // ofstream is not recognized in mac
// #endif

// #include <fmt/core.h>
#include <yaml-cpp/yaml.h>

// * define the demention of check board
#define NUM_HORIZON_CORNER  10
#define NUM_VERTICAL_CORNER 7
#define NUM_SQUARE_LEN      25 // 25 mm

#define FPS 30 // 30 fps

using namespace cv;
using namespace std::chrono;

static const std::string CALIB_CONFIG_DIR_PATH = std::string(CONFIG_PATH);
 
int main()
{
    // std::cout << fmt::format("hello world") << std::endl;

    auto cap = VideoCapture(0);

    cap.set(CAP_PROP_FRAME_WIDTH, 660);
    cap.set(CAP_PROP_FRAME_HEIGHT, 660);
    std::cout << "width  : " << cap.get(CAP_PROP_FRAME_WIDTH) << std::endl;
    std::cout << "height : " << cap.get(CAP_PROP_FRAME_HEIGHT) << std::endl;

    // 640(width) 640(height) 30(fps)


    if (!cap.isOpened()) {
        throw std::runtime_error(
            "cap is not opened."
        );
    }

    // Creating vector to store vectors of 3D points for each checkerboard image
    std::vector<std::vector<Point3f>> objpoints;
 
    // Creating vector to store vectors of 2D points for each checkerboard image
    std::vector<std::vector<Point2f>> imgpoints;

    Mat cameraMatrix;
    Mat distCoeffs;
    Mat R;
    Mat T;

    Mat image_;
    while (true) {
        auto start_time = high_resolution_clock::now();

        static bool first_calib = false;

        static int capture_cnt = 0;
        capture_cnt += 1;

        // * ----- START calculation START -----
        if (!cap.read(image_)) {
            throw std::runtime_error(
                "End of video Or cannot get image from VideoCapture"
            );
        }

        Mat image;
        if (first_calib) {
            undistort(image_, image, cameraMatrix, distCoeffs);
        } else {
            image = image_;
        }

        // Defining the world coordinates for 3D points
        std::vector<cv::Point3f> objp;
        for(int i = 0; i < NUM_HORIZON_CORNER; i++) {
            for(int j = 0; j < NUM_VERTICAL_CORNER; j++) {
                objp.push_back(cv::Point3f(j * NUM_SQUARE_LEN, i * NUM_SQUARE_LEN, 0.0f));
            }
        }

        Mat gray;
        cv::cvtColor(image, gray, COLOR_BGR2GRAY);

        std::vector<Point2f> corner_pts;

        bool success = cv::findChessboardCorners(
            gray,
            cv::Size(NUM_VERTICAL_CORNER, NUM_HORIZON_CORNER),
            corner_pts,
            CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE
        );

        std::cout << "success: " << std::boolalpha << success << std::endl;

        if (success) {
            cv::TermCriteria criteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.001);
       
            // refining pixel coordinates for given 2d points.
            cv::cornerSubPix(gray, corner_pts,cv::Size(11, 11), cv::Size(-1, -1), criteria);
       
            // Displaying the detected corner points on the checker board
            cv::drawChessboardCorners(image, cv::Size(NUM_VERTICAL_CORNER, NUM_HORIZON_CORNER), corner_pts, success);
       
            if (capture_cnt > 30 * 3) { // caputre every 3 sec
                objpoints.push_back(objp);
                imgpoints.push_back(corner_pts);
                capture_cnt = 0;

                calibrateCamera(objpoints, imgpoints, cv::Size(gray.rows, gray.cols), cameraMatrix, distCoeffs, R, T);

                first_calib = true;
            }
        }

        auto end_time = high_resolution_clock::now();
        duration<double> elapse = end_time - start_time;

        double desired_sleep_time = 1.0 / (double) FPS;
        double sleep_time = desired_sleep_time - elapse.count();

        if (sleep_time > 0) {
            std::this_thread::sleep_for(microseconds(static_cast<int>(sleep_time * 1e6)));
        }

        imshow("Camera img", image);

        // * -----  END  calculation  END  -----
        
        if (waitKey(1) == 27) {
            std::cout << "Calibration finished!" << std::endl;
            break;
        } 
    }

    std::cout << "cameraMatrix: " << cameraMatrix << std::endl;
    std::cout << "distCoeff: " << distCoeffs << std::endl;
    std::cout << "Rotation: " << R << std::endl;
    std::cout << "Translation: " << T << std::endl;


    YAML::Node calib_config;

    calib_config["camera_matrix"]["rows"] = cameraMatrix.rows;
    calib_config["camera_matrix"]["cols"] = cameraMatrix.cols;

    // YAML::Node calib_mx_data_node;
    std::vector<double> cam_mx_data_vec((double*) cameraMatrix.datastart, (double*) cameraMatrix.dataend);
    // calib_mx_data_node.push_back(cam_mx_data_vec);
    calib_config["camera_matrix"]["data"] = cam_mx_data_vec;

    calib_config["dist_coeffs"]["rows"] = distCoeffs.rows;
    calib_config["dist_coeffs"]["cols"] = distCoeffs.cols;

    // YAML::Node dist_coeff_data_node;
    std::vector<double> dist_coef_data_vec((double*) distCoeffs.datastart, (double*) distCoeffs.dataend);
    // dist_coeff_data_node.push_back(dist_coef_data_vec);

    calib_config["dist_coeffs"]["data"] = dist_coef_data_vec;


    // /** ----------------------- projection matrix ----------------------- */
    Mat outputRotation;
    Rodrigues(R.row(R.rows - 1), outputRotation);
    Mat Rt;
    hconcat(outputRotation, T.row(T.rows - 1), Rt);

    // // ! PROJECTION MX
    Mat projection_mx;
    gemm(cameraMatrix, Rt, 1.0, Mat(), 0.0, projection_mx);

    calib_config["projection_matrix"]["rows"] = projection_mx.rows;
    calib_config["projection_matrix"]["cols"] = projection_mx.cols;
    // // calib_config["projection_matrix"]["data"] = projection_mx.data;

    std::vector<double> proj_mx_vec((double*) projection_mx.datastart, (double*) projection_mx.dataend);

    calib_config["projection_matrix"]["data"] = proj_mx_vec;

    remove((CALIB_CONFIG_DIR_PATH + "/calib_config.yaml").c_str());

    std::ofstream yaml_fout(CALIB_CONFIG_DIR_PATH + "/calib_config.yaml");
    yaml_fout << YAML::Dump(calib_config);
    yaml_fout.close();

    cap.release();

    return 0;
}