
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>

#define KERNEL_SIZE 65.0
#define LEN 35.0
#define ANGLE_DEGREES 15.0

void circleShift(cv::Mat &src, cv::Mat &dst, int dx, int dy)
{
    dst = cv::Mat::zeros(src.size(), src.type());
    int rows = src.rows;
    int cols = src.cols;

    dx = ((dx % rows) + rows) % rows;
    dy = ((dy % cols) + cols) % cols;

    // top-left
    src(cv::Rect(0, 0, cols - dy, rows - dx)).copyTo(dst(cv::Rect(dy, dx, cols - dy, rows - dx)));
    // top-right
    src(cv::Rect(cols - dy, 0, dy, rows - dx)).copyTo(dst(cv::Rect(0, dx, dy, rows - dx)));
    // bottom-left
    src(cv::Rect(0, rows - dx, cols - dy, dx)).copyTo(dst(cv::Rect(dy, 0, cols - dy, dx)));
    // bottom-right
    src(cv::Rect(cols - dy, rows - dx, dy, dx)).copyTo(dst(cv::Rect(0, 0, dy, dx)));
}

cv::Mat multiplyComplex(const cv::Mat &A, const cv::Mat &B)
{
    cv::Mat C;
    cv::mulSpectrums(A, B, C, 0, false);
    return C;
}

cv::Mat complexToImageIDFT(const cv::Mat &complex)
{
    cv::Mat inv;
    idft(complex, inv, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
    return inv;
}

cv::Mat createMotionPSF()
{
    cv::Mat psf = cv::Mat::zeros(cv::Size(KERNEL_SIZE, KERNEL_SIZE), CV_32F);
    cv::Point center = cv::Point(KERNEL_SIZE - 1 / 2, KERNEL_SIZE - 1 / 2);

    double angle = ANGLE_DEGREES * CV_PI / 180.0;

    double dx = cos(angle) * (LEN - 1) / 2.0;
    double dy = sin(angle) * (LEN - 1) / 2.0;

    cv::Point p1 = cv::Point(cvRound(center.x - dx), cvRound(center.y - dy));
    cv::Point p2 = cv::Point(cvRound(center.x + dx), cvRound(center.y + dy));

    cv::line(psf, p1, p2, cv::Scalar(1.0), 1, cv::LINE_AA);

    cv::Scalar s = cv::sum(psf);
    if (s[0] != 0)
        psf /= s[0];

    return psf;
}

cv::Mat imageToComplexDFT(cv::Mat &img_float)
{
    std::vector<cv::Mat> planes = {img_float.clone(), cv::Mat::zeros(img_float.size(), CV_32F)};
    cv::Mat complex;
    cv::merge(planes, complex);
    dft(complex, complex, 0);
    return complex;
}

cv::Mat psf2otf(cv::Mat &psf, cv::Size size)
{
    cv::Mat psf_padded = cv::Mat::zeros(size, CV_32F);

    cv::Mat roi(psf_padded, cv::Rect(0, 0, psf.cols, psf.rows));
    psf.convertTo(roi, CV_32F);

    unsigned cx = psf.rows / 2;
    unsigned cy = psf.cols / 2;

    cv::Mat shifted;
    circleShift(psf_padded, shifted, -cx, -cy);

    cv::Mat otf;
    dft(shifted, otf, cv::DFT_COMPLEX_OUTPUT);
    return otf;
}

cv::Mat conjugateComplex(const cv::Mat &A)
{
    cv::Mat B = cv::Mat::zeros(A.size(), A.type());
    std::vector<cv::Mat> planes(2), outPlanes(2);
    split(A, planes);
    outPlanes[0] = planes[0];
    outPlanes[1] = -planes[1];
    merge(outPlanes, B);
    return B;
}

cv::Mat magnitudeSquared(const cv::Mat &complex)
{
    std::vector<cv::Mat> p(2);
    split(complex, p);
    cv::Mat mag2;
    magnitude(p[0], p[1], mag2);
    mag2 = mag2.mul(mag2);
    return mag2;
}

cv::Mat wienerDeconvolution(cv::Mat &blurredFloat, cv::Mat &psf, double K)
{
    cv::Size size = blurredFloat.size();
    cv::Mat G = imageToComplexDFT(blurredFloat);
    cv::Mat H = psf2otf(psf, size);
    cv::Mat Hconj = conjugateComplex(H);
    cv::Mat Hmag2 = magnitudeSquared(H);

    std::vector<cv::Mat> Hc_planes(2);
    split(Hconj, Hc_planes);

    cv::Mat denom = Hmag2 + K;
    cv::Mat Wr, Wi;
    divide(Hc_planes[0], denom, Wr);
    divide(Hc_planes[1], denom, Wi);

    cv::Mat W;
    cv::Mat Wplanes[] = {Wr, Wi};
    merge(Wplanes, 2, W);

    cv::Mat F_est = multiplyComplex(G, W);

    cv::Mat f_est = complexToImageIDFT(F_est);
    return f_est;
};

cv::Mat applyPSF(cv::Mat &img_float, cv::Mat &psf)
{
    cv::Size size = img_float.size();
    cv::Mat img_dft = imageToComplexDFT(img_float);
    cv::Mat otf = psf2otf(psf, size);

    cv::Mat res_spec = multiplyComplex(img_dft, otf);
    cv::Mat res = complexToImageIDFT(res_spec);

    return res;
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " <input_image>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string input_file = argv[1];

    cv::Mat input_image = cv::imread(input_file, cv::IMREAD_COLOR);

    if (input_image.empty())
    {
        std::cerr << "Failed to read input image: " << input_file << std::endl;
        return EXIT_FAILURE;
    }

    cv::Mat img_gray;
    cv::cvtColor(input_image, img_gray, cv::COLOR_BGR2GRAY);

    cv::Mat img_float;
    img_gray.convertTo(img_float, CV_32F, 1.0 / 255.0);

    cv::Mat psf = createMotionPSF();

    cv::Mat blurred = applyPSF(img_float, psf);

    cv::Mat noise = cv::Mat(blurred.size(), CV_32F);
    cv::randn(noise, 0.0f, 0.005f);
    cv::Mat blurredNoisy = blurred + noise;
    blurredNoisy = cv::max(0.0f, cv::min(1.0f, blurredNoisy));

    double k = 0.001;
    cv::Mat deconvoled = wienerDeconvolution(blurredNoisy, psf, k);
    deconvoled = cv::max(0.0f, cv::min(1.0f, deconvoled));

    cv::Mat blurred8U, deconvoled8U;
    blurredNoisy.convertTo(blurred8U, CV_8U, 255.0);
    deconvoled.convertTo(deconvoled8U, CV_8U, 255.0);

    cv::imwrite("blurred.png", blurred8U);
    cv::imwrite("deconvoled.png", deconvoled8U);

    cv::Mat psfVis;
    cv::normalize(psf, psfVis, 0, 255, cv::NORM_MINMAX, CV_8U);
    psfVis.convertTo(psfVis, CV_8U);

    cv::imwrite("psf.png", psfVis);

    cv::imshow("Input Image", input_image);
    cv::imshow("Blurred Image", blurred8U);
    cv::imshow("Deconvoled Image", deconvoled8U);
    cv::waitKey(0);

    std::cout << "Done." << std::endl;

    return EXIT_SUCCESS;
}


// g++ main.cpp -o psf_main `pkg-config --cflags --libs opencv4` -std=c++17
