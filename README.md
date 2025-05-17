# 🔥💨 Fire Smoke Detection & Telegram Alert System

## 📖 Overview  
This project is a real-time fire and smoke detection system that leverages YOLOv11 for object detection and OpenCV for video stream processing. 
The system continuously analyzes video streams (from a webcam or saved footage), detects fire and smoke, draws bounding boxes with labels, and sends annotated snapshots with alerts to a Telegram chat. It is designed to provide early warnings for fire hazards, ensuring quick response and safety.

---

## 🚀 Features  
- **Real-Time Detection**: Detects fire and smoke in video streams with high accuracy using YOLOv11.
- **Motion Detection**: Filters frames based on motion detection to reduce false positives. 
- **Alert System**: Sends real-time alerts with annotated images to a Telegram chat.
- **Configurable Sensitivity**: Adjust detection confidence threshold and notification frequency. 
- **Visualization**: Displays the detection results with bounding boxes and labels on the video feed.

---

## 🛠 Technologies & Libraries  
- **Model & Inference**:  
  - YOLOv11 (Ultralytics)
  - PyTorch, OpenCV  
- **Notification**:  
  - TelegramBot API 
- **Data & Annotation**:  
  - Roboflow for dataset labeling  
- **Environment**:  
  - Python 3.8+  
  - Google Colab (training)  

---

## ⚙️ Installation & Setup  

1️⃣ **Training the Model (Google Colab)**

- Open the Colab Notebook & Run the Fire_detection_Model.ipynb file

- Download the best.pt file to your local machine for deployment.

2️⃣ **Running the Detection Script Locally**

- Setup Local Environment: Ensure you have Python 3.8+ installed.

- Install required libraries:
  
```bash
pip install python-dotenv opencv-python
```

- Prepare Environment Variables

  * Create a `.env` file in the project root directory with the following content:

```plaintext
TELEGRAM_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```


- Run the Detection Script
```bash
python FireSmokeDetection.py
```
- Make sure the `VIDEO_PATH` in the script points to the correct video file.
  
3️⃣ **System Workflow**

- Video Input: Reads video input from a file or webcam.

- Detection: YOLOv11 detects fire and smoke in the frames, draws bounding boxes, and adds labels.

- Alert System: Sends annotated frames to Telegram if fire or smoke is detected.

- User Control: Send /stop to the bot to terminate detection.

4️⃣ **Notes**

- GPU in Colab: Ensure you enable GPU acceleration (Runtime > Change runtime type > GPU).

---

## 👤 Author
Developed by **THUC TU**

---

## Contact

For any questions or feedback, please contact:

* **Email**: [tuthucdz@gmail.com](mailto:tuthucdz@gmail.com)
* **GitHub**: [Ne4nf](https://github.com/Ne4nf)
