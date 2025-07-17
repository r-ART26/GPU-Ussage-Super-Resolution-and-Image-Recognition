#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// Función para cargar las etiquetas desde un archivo
std::vector<std::string> loadLabels(const std::string& path) {
    std::ifstream file(path);
    std::vector<std::string> labels;
    std::string line;
    if (!file.is_open()) {
        throw std::runtime_error("No se pudo abrir el archivo de etiquetas.");
    }
    while (std::getline(file, line)) {
        if (!line.empty()) labels.push_back(line);
    }
    return labels;
}

// Función auxiliar para aplicar la función sigmoide
cv::Mat applySigmoid(const cv::Mat& input) {
    cv::Mat output;
    cv::exp(-input, output);
    return 1.0 / (1.0 + output);
}

// Configura el backend del modelo según la opción -gpu o -cpu
void configureBackend(cv::dnn::Net& net, bool gpu) {
    if (gpu) {
        try {
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
            std::cout << "Usando backend CUDA.\n";
        } catch (...) {
            std::cerr << "CUDA no disponible. Se usará CPU.\n";
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        }
    } else {
        std::cout << "Usando backend CPU.\n";
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
}

// Realiza inferencia y devuelve las etiquetas con probabilidad alta
std::string classifyFrame(cv::dnn::Net& net, const cv::Mat& frame, const std::vector<std::string>& classNames, float threshold = 0.3f, int topK = 2) {
    cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0 / 255.0, cv::Size(224, 224), cv::Scalar(0.485, 0.456, 0.406), true, false);
    net.setInput(blob);
    cv::Mat output = applySigmoid(net.forward());

    std::vector<std::pair<int, float>> predictions;
    for (int i = 0; i < output.cols; ++i) {
        float prob = output.at<float>(0, i);
        if (prob > threshold) {
            predictions.emplace_back(i, prob);
        }
    }

    std::sort(predictions.begin(), predictions.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });

    if ((int)predictions.size() > topK) predictions.resize(topK);

    std::string label;
    if (predictions.empty()) {
        label = cv::format("Ninguna (>%.2f)", threshold);
    } else {
        for (size_t i = 0; i < predictions.size(); ++i) {
            int idx = predictions[i].first;
            float p = predictions[i].second;
            std::string name = idx < (int)classNames.size() ? classNames[idx] : "Clase" + std::to_string(idx);
            label += cv::format("%s(%.2f)", name.c_str(), p);
            if (i + 1 < predictions.size()) label += ", ";
        }
    }
    return label;
}

// Dibuja texto sobre un frame con fondo negro
void drawTextWithBackground(cv::Mat& frame, const std::string& text, cv::Point origin, cv::Scalar color) {
    int baseLine;
    cv::Size size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseLine);
    cv::rectangle(frame, origin + cv::Point(0, baseLine), origin + cv::Point(size.width, -size.height), cv::Scalar(0, 0, 0), cv::FILLED);
    cv::putText(frame, text, origin, cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 1);
}

int main(int argc, char** argv) {
    bool useGPU = std::any_of(argv + 1, argv + argc, [](const char* arg) {
        return std::strcmp(arg, "-gpu") == 0;
    });

    std::vector<std::string> classNames;
    try {
        classNames = loadLabels("Labels/imagenet_classes.txt");
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }

    cv::dnn::Net model = cv::dnn::readNetFromONNX("Models/mobilenetv3-small-100.onnx");
    configureBackend(model, useGPU);

    cv::VideoCapture camera("http://192.168.0.101:8080/video");
    if (!camera.isOpened()) {
        std::cerr << "No se pudo abrir el stream.\n";
        return 1;
    }

    std::string label = "Esperando...";
    int frameCounter = 0;
    double fps = 0.0;
    auto lastFpsUpdate = std::chrono::steady_clock::now();
    auto lastInference = lastFpsUpdate - std::chrono::seconds(2);

    cv::Mat frame;
    while (true) {
        camera >> frame;
        if (frame.empty()) break;
        frameCounter++;

        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - lastFpsUpdate).count() >= 1000) {
            fps = frameCounter * 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(now - lastFpsUpdate).count();
            frameCounter = 0;
            lastFpsUpdate = now;
        }

        if (now - lastInference >= std::chrono::seconds(2)) {
            label = classifyFrame(model, frame, classNames);
            lastInference = now;
        }

        drawTextWithBackground(frame, label, {10, 30}, cv::Scalar(0, 255, 0));
        drawTextWithBackground(frame, cv::format("FPS: %.1f", fps), {frame.cols - 150, 30}, cv::Scalar(255, 255, 0));

        cv::imshow("Clasificación Multietiqueta", frame);
        if (cv::waitKey(1) == 27) break;
    }

    camera.release();
    cv::destroyAllWindows();
    return 0;
}
