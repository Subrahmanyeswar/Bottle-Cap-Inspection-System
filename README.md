# 🏭 Edge AI Quality Inspection System

Advanced Edge-Based Automated Optical Inspection System for Bottle Manufacturing using Optimized MobileNetV2.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg) ![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg) ![OpenVINO](https://img.shields.io/badge/OpenVINO-Optimized-blueviolet.svg) ![TensorFlow](https://img.shields.io/badge/TensorFlow-Lite-orange.svg) ![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🚀 Features

* 🤖 **Edge AI Processing** - Runs entirely on standard CPU hardware without needing expensive GPUs.
* ⚡ **Ultra-Fast Inference** - Achieves **447 FPS** with just 2.23ms latency using Intel OpenVINO optimization.
* 🎯 **High Accuracy** - **95% overall classification accuracy** with 100% recall on critical defects (Missing Caps).
* 🎥 **Smart Motion Trigger** - Uses frame differencing to reduce idle CPU overhead by 60%.
* 💡 **Adaptive Preprocessing** - Implements CLAHE for robust performance across varying factory lighting conditions (200-2000 lux).

## 📸 System States

The inspection pipeline dynamically switches between an idle monitoring state and active classification, detecting three specific bottle cap conditions.

| 1. Idle State (Waiting for Bottle) | 2. Active Inspection (Pass / Warning / Critical) |
| :---: | :---: |
| <img src="idle_state.png" width="400"> | <img src="all_3_states.png" width="400"> |

* **PASS (Green):** Properly sealed caps.
* **WARNING (Orange):** Loose or crooked caps.
* **CRITICAL (Red):** Missing caps entirely.

## 🛠️ Hardware Requirements

* **Processor**: Standard Laptop CPU (Benchmarked on Intel Core i5-12500HX)
* **Camera**: Integrated 720p HD Webcam
* **RAM**: 16 GB DDR4 (Minimum 8GB Supported)

## 💻 Installation & Usage

**1. Clone the repository**
```bash
git clone [https://github.com/Subrahmanyeswar/Bottle-Cap-Inspection-System.git](https://github.com/Subrahmanyeswar/Bottle-Cap-Inspection-System.git)
cd Bottle-Cap-Inspection-System
2. Install dependencies

Bash
pip install -r requirements.txt
3. Run the inspection system

Bash
python main.py
📊 Performance Benchmark
Our system demonstrates a 23.8% performance improvement when migrating from standard TensorFlow Lite (FP32) to Intel OpenVINO (FP16). Model size was successfully compressed by 48.3% (from 8.9 MB to 4.6 MB) with zero loss to critical defect detection.

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
Neural Network Architecture based on MobileNetV2 (Sandler et al.)

Accelerated inference powered by Intel OpenVINO Toolkit

Computer Vision pipeline built with OpenCV

⭐ Star this repo if you found it helpful!
Made with ❤️ and 🤖 by Subrahmanyeswar Kolluru

👨‍💻 Author
Subrahmanyeswar Kolluru

GitHub: https://github.com/Subrahmanyeswar

LinkedIn: www.linkedin.com/in/subrahmanyeswar-kolluru-914694293