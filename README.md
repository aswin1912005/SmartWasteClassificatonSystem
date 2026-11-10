# ♻️ Smart Waste Classification System

### AI-Powered Automatic Waste Segregation Using Deep Learning and IoT

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch-orange?logo=pytorch)
![Flask](https://img.shields.io/badge/Backend-Flask-black?logo=flask)
![ESP32](https://img.shields.io/badge/Hardware-ESP32-red)
![IoT](https://img.shields.io/badge/Technology-IoT-green)
![Status](https://img.shields.io/badge/Project-Completed-success)

---

## 🌱 1. Project Overview

The **Smart Waste Classification System** is an AI-powered waste segregation solution designed mainly for homes, offices, and small institutions.

The system automatically identifies waste materials and classifies them into two major categories:

- 🍃 **Biodegradable Waste**
- 🧴 **Non-Biodegradable Waste**

It combines deep learning, embedded hardware, image processing, and web technology to reduce manual effort and improve waste segregation.

The project uses the MobileNetV3-Small deep learning model for image classification and ESP32-based hardware for automatic waste sorting.

## 🎯 2. Objectives

- Develop an automatic waste classification system.
- Identify biodegradable and non-biodegradable waste.
- Apply deep learning for image-based waste recognition.
- Reduce manual waste segregation.
- Automate waste sorting using a servo motor.
- Display captured images and classification results through a web dashboard.
- Provide a compact and affordable solution for home and office use.

## ✨ 3. Key Features

- 🤖 AI-based waste classification using MobileNetV3-Small.
- 📸 Automatic image capture using ESP32-CAM.
- 🔍 Waste detection using an IR sensor.
- ⚙️ Automatic sorting using an MG996R servo motor.
- 🌐 Flask-based image processing and model inference.
- 📊 Web dashboard for displaying images and classification results.
- 🏠 Compact design for homes and offices.
- 💾 Local file storage without requiring a database.

## 🏗️ 4. System Architecture

The system consists of three major modules:

1. Object Detection and Image Capture
2. AI-Based Waste Classification
3. Automatic Waste Segregation

### System Workflow

```text
        IR Sensor
            |
            v
     ESP32 Controller
            |
      HTTP Trigger
            |
            v
        ESP32-CAM
            |
      Capture Image
            |
            v
       Flask Server
            |
            v
    MobileNetV3-Small
            |
            v
   Waste Classification
            |
       +----+----+
       |         |
       v         v
 Biodegradable  Non-Biodegradable
       |         |
       +----+----+
            |
     Classification Result
            |
            v
     ESP32 Controller
            |
            v
      MG996R Servo
            |
            v
    Automatic Sorting
            |
            v
      Appropriate Bin

       Web Dashboard
    Displays Image and Result
```

## 🔄 5. Working Principle

**Step 1 – Waste Detection**

The IR sensor detects the presence of waste and sends a signal to the ESP32 controller.

**Step 2 – Image Capture**

The ESP32 controller directly triggers the ESP32-CAM to capture an image of the waste.

**Step 3 – Image Transmission**

The ESP32-CAM sends the captured image to the Flask server through Wi-Fi.

**Step 4 – AI Classification**

The Flask server processes the image using the trained MobileNetV3-Small model and predicts the waste category.

**Step 5 – Waste Segregation**

The classification result is returned to the ESP32 controller, which operates the MG996R servo motor to direct the waste into the appropriate bin.

**Step 6 – Web Dashboard**

The web dashboard displays the captured image and its corresponding classification result.

## 🛠️ 6. Hardware Requirements

| Component | Purpose |
|---|---|
| ESP32 DevKit V1 | Main controller |
| ESP32-CAM | Waste image capture |
| IR Sensor | Object detection |
| MG996R Servo Motor | Automatic waste sorting |
| Breadboard | Circuit connections |
| Jumper Wires | Electrical connections |
| 12V DC Adapter | Power supply |
| Buck Converter | Voltage step-down |
| USB-to-TTL Adapter | ESP32-CAM programming |

## 💻 7. Software Requirements

| Software / Technology | Purpose |
|---|---|
| Python | Backend programming |
| Flask | Web server and image processing |
| PyTorch | Deep learning framework |
| MobileNetV3-Small | Waste classification |
| HTML | Dashboard structure |
| CSS | Dashboard styling |
| JavaScript | Dashboard functionality |
| Arduino IDE | ESP32 programming |
| OpenCV / Pillow | Image processing, as required |

## 🧠 8. AI Model Details

The project uses **MobileNetV3-Small**, a lightweight convolutional neural network designed for efficient image classification.

### Dataset Information

| Parameter | Details |
|---|---|
| Dataset Size | 4,000 Images |
| Model | MobileNetV3-Small |
| Framework | PyTorch |
| Output Categories | Biodegradable and Non-Biodegradable |
| Input | Captured Waste Image |
| Output | Predicted Waste Category |

### Classification Categories

| Category | Example Waste |
|---|---|
| Biodegradable | Banana peels, organic waste |
| Non-Biodegradable | Plastic covers, metal objects |

The trained model analyzes visual features such as shape, colour, texture, and appearance to identify waste materials.

## 🌐 9. Web Dashboard

The web dashboard provides a simple interface for monitoring the waste classification process.

### Dashboard Features

- Display captured waste images.
- Show predicted waste categories.
- View the latest classification results.
- Monitor waste identification output.
- Provide a simple and user-friendly interface.

**Technologies:** HTML, CSS, JavaScript, Python, and Flask.

## 📂 10. Project Structure

The following is a suggested project structure. Adjust filenames and folders to match your actual repository.

```text
Smart-Waste-Classification/
│
├── dataset/
│   ├── train/
│   ├── test/
│   └── validation/
│
├── model/
│   └── best_model.pth
│
├── templates/
│   └── index.html
│
├── static/
│   ├── css/
│   ├── js/
│   └── images/
│
├── uploads/
│
├── esp32/
│   ├── esp32_controller.ino
│   └── esp32_cam.ino
│
├── app.py
├── requirements.txt
└── README.md
```

## ⚙️ 11. Installation and Setup

### Prerequisites

Make sure the following are installed:

- Python 3.x
- Arduino IDE
- Git
- ESP32 Board Package for Arduino IDE

### Step 1 – Clone the Repository

```bash
git clone https://github.com/your-username/Smart-Waste-Classification.git
```

### Step 2 – Open the Project Folder

```bash
cd Smart-Waste-Classification
```

### Step 3 – Create a Virtual Environment

```bash
python -m venv venv
```

### Step 4 – Activate the Virtual Environment

Windows:

```bash
venv\Scripts\activate
```

Linux / Ubuntu:

```bash
source venv/bin/activate
```

### Step 5 – Install Dependencies

```bash
pip install -r requirements.txt
```

If the requirements file is not available, install the required packages according to your implementation:

```bash
pip install flask torch torchvision pillow opencv-python
```

### Step 6 – Configure the Project

- Place the trained model file in the model directory.
- Update the model path in the Flask application.
- Configure the Wi-Fi SSID and password in the ESP32 code.
- Update the ESP32-CAM and Flask server IP addresses.
- Verify the hardware connections and power supply.

### Step 7 – Run the Flask Server

```bash
python app.py
```

Open the dashboard in your browser using the local server address configured in the Flask application.

### Step 8 – Upload ESP32 Programs

- Upload the controller program to ESP32 DevKit V1.
- Upload the camera program to ESP32-CAM.
- Configure the devices to communicate over the same Wi-Fi network.
- Start the system and test waste detection.

## 🧪 12. Testing

The system can be tested using different waste materials.

| Test Input | Expected Category |
|---|---|
| Banana Peel | Biodegradable |
| Paper | Biodegradable |
| Plastic Cover | Non-Biodegradable |
| Metal Object | Non-Biodegradable |

Testing includes:

- IR sensor detection.
- ESP32 and ESP32-CAM communication.
- Image transmission to Flask.
- AI-based image classification.
- Servo motor sorting operation.
- Web dashboard output.

Actual test results should be recorded based on observed system performance.

## 📸 13. Project Screenshots

Add your actual project images to a `screenshots` folder.

```markdown
![Hardware Setup](screenshots/hardware-setup.jpg)

![Web Dashboard](screenshots/dashboard.png)

![Waste Detection](screenshots/waste-detection.png)

![Classification Result](screenshots/classification.png)

![Automatic Sorting](screenshots/automatic-sorting.jpg)
```

## 🚀 14. Future Enhancements

- Improve classification accuracy with a larger dataset.
- Add more waste categories such as glass and paper.
- Introduce remote IoT monitoring.
- Add ultrasonic sensors for bin-level detection.
- Develop a mobile application.
- Improve image processing speed.
- Integrate solar power.

## 🎯 15. Applications

- 🏠 Residential Buildings
- 🏢 Offices
- 🏫 Educational Institutions
- 🏬 Small Commercial Spaces
- 🏘️ Community Waste Management

## 🌍 16. Benefits

- Reduces manual waste segregation.
- Improves waste identification.
- Supports automatic waste sorting.
- Saves time and human effort.
- Encourages proper waste disposal.
- Promotes a cleaner environment.

## 👨‍💻 17. Author

**Aswin C**

MCA Student

**Project:** Smart Waste Classification System

---

## ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.

### ♻️ Smart Waste Classification System

**Smart Technology for a Cleaner Environment! 🌍**

