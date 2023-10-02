#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <image-path>" << std::endl;
        return -1;
    }

    cv::Mat img = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Error: Unable to load image: " << argv[1] << std::endl;
        return -1;
    }

    cv::HOGDescriptor hog;
    hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());

    std::vector<cv::Rect> detections;
    hog.detectMultiScale(img, detections);

    for(const auto& rect : detections) {
        cv::rectangle(img, rect, cv::Scalar(0, 255, 0), 2);
    }

    cv::imshow("Pedestrian Detection", img);
    cv::waitKey(0);

    return 0;
}

