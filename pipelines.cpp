#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>

using namespace std;
using namespace cv;

// Timer sencillo para medir tiempos
class Timer {
    TickMeter tm;
public:
    void start() { tm.reset(); tm.start(); }
    void stop()  { tm.stop(); }
    double get() const { return tm.getTimeMilli(); }
};

// Estructura para manejar filtros GPU
struct GPUFilters {
    Ptr<cuda::Filter> gaussian;
    Ptr<cuda::Filter> erode;
    Ptr<cuda::Filter> dilate;
    Ptr<cuda::CannyEdgeDetector> canny;

    GPUFilters() {
        gaussian = cuda::createGaussianFilter(CV_8UC1, CV_8UC1, Size(5, 5), 1.5);
        erode    = cuda::createMorphologyFilter(MORPH_ERODE, CV_8UC1,
                      getStructuringElement(MORPH_RECT, Size(3, 3)));
        dilate   = cuda::createMorphologyFilter(MORPH_DILATE, CV_8UC1,
                      getStructuringElement(MORPH_RECT, Size(3, 3)));
        canny    = cuda::createCannyEdgeDetector(50.0, 150.0);
    }
};

// Inicializa fuente de video o imagen
bool initInput(const string& inputPath, VideoCapture& cap, Mat& image, bool& isVideo) {
    if (inputPath.empty()) {
        cap.open(0);
    } else {
        cap.open(inputPath);
    }

    if (cap.isOpened()) {
        isVideo = true;
        return true;
    }

    image = imread(inputPath, IMREAD_GRAYSCALE);
    if (!image.empty()) {
        isVideo = false;
        return true;
    }

    cerr << "No se pudo abrir el archivo o cámara." << endl;
    return false;
}

// Verifica disponibilidad de GPU
bool initCUDA() {
    if (cuda::getCudaEnabledDeviceCount() == 0) {
        cerr << "No se encontraron dispositivos CUDA." << endl;
        return false;
    }
    cuda::setDevice(0);
    return true;
}

// Pipeline en CPU
void processCPU(const Mat& gray, Mat& edges, Mat& histEq) {
    Mat blur, eroded, dilated;
    GaussianBlur(gray, blur, Size(5, 5), 1.5);
    erode(blur, eroded, getStructuringElement(MORPH_RECT, Size(3, 3)));
    dilate(eroded, dilated, getStructuringElement(MORPH_RECT, Size(3, 3)));
    Canny(dilated, edges, 50, 150);
    equalizeHist(gray, histEq);
}

// Pipeline en GPU
void processGPU(const Mat& gray, GPUFilters& filters, Mat& edges, Mat& histEq) {
    cuda::GpuMat d_gray(gray), d_gauss, d_erode, d_dilate, d_edges, d_eq;
    filters.gaussian->apply(d_gray, d_gauss);
    filters.erode->apply(d_gauss, d_erode);
    filters.dilate->apply(d_erode, d_dilate);
    filters.canny->detect(d_dilate, d_edges);
    cuda::equalizeHist(d_gray, d_eq);
    d_edges.download(edges);
    d_eq.download(histEq);
}

// Muestra los resultados lado a lado
void displayResults(const Mat& cpuEdges, const Mat& gpuEdges, const Mat& cpuEq, const Mat& gpuEq) {
    Mat top, bottom;
    hconcat(cpuEdges, gpuEdges, top);
    hconcat(cpuEq, gpuEq, bottom);
    imshow("Edges: CPU | GPU", top);
    imshow("Hist EQ: CPU | GPU", bottom);
}

int main(int argc, char* argv[]) {
    string inputPath = (argc >= 2) ? argv[1] : "";
    VideoCapture cap;
    Mat image;
    bool isVideo = false;

    if (!initInput(inputPath, cap, image, isVideo) || !initCUDA()) return -1;

    GPUFilters filters;
    Timer cpuTimer, gpuTimer;
    int frameCount = 0;
    double totalCPU = 0, totalGPU = 0;

    while (true) {
        Mat frame, gray;
        if (isVideo) {
            if (!cap.read(frame)) break;
            cvtColor(frame, gray, COLOR_BGR2GRAY);
        } else {
            gray = image.clone();  // en caso de imagen fija
        }

        // CPU
        Mat cpuEdges, cpuEq;
        cpuTimer.start();
        processCPU(gray, cpuEdges, cpuEq);
        cpuTimer.stop();

        // GPU
        Mat gpuEdges, gpuEq;
        gpuTimer.start();
        processGPU(gray, filters, gpuEdges, gpuEq);
        gpuTimer.stop();

        totalCPU += cpuTimer.get();
        totalGPU += gpuTimer.get();
        frameCount++;

        displayResults(cpuEdges, gpuEdges, cpuEq, gpuEq);

        if (waitKey(1) == 27 || !isVideo) break;
    }

    cout << "\nFrames procesados: " << frameCount << endl;
    cout << "Tiempo promedio CPU: " << (totalCPU / frameCount) << " ms\n";
    cout << "Tiempo promedio GPU: " << (totalGPU / frameCount) << " ms\n";

    return 0;
}
