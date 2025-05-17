import os
import time
import cv2
import telebot
import threading
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv() 

TOKEN    = os.getenv("TELEGRAM_TOKEN")
CHAT_ID  = os.getenv("TELEGRAM_CHAT_ID")
bot     = telebot.TeleBot(TOKEN)


MODEL_PATH = r"D:\best (1).pt"
VIDEO_PATH = r"D:\Fire_Smoke_Detector\Input\smoke01.mp4" 
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Không tìm thấy mô hình tại {MODEL_PATH}")
if not os.path.exists(VIDEO_PATH):
    raise FileNotFoundError(f"Không tìm thấy video tại {VIDEO_PATH}")

model = YOLO(MODEL_PATH)

running             = True
fire_alert_count    = 0
smoke_alert_count   = 0
MAX_ALERTS          = 3
ALERT_INTERVAL      = 5.0  
last_fire_alert     = 0.0
last_smoke_alert    = 0.0
MOTION_THRESHOLD    = 0.8 
prev_frame_gray     = None   


cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Không thể mở video!")



def draw_label(frame, x1, y1, x2, y2, conf, text, color):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)
    label   = f"{text}: {conf:.2f}"
    font    = cv2.FONT_HERSHEY_SIMPLEX
    font_sc = 0.5
    font_th = 2
    (tw, th), baseline = cv2.getTextSize(label, font, font_sc, font_th)
    pad = 4
    ly  = y1 - pad
    if ly - th - baseline < 0:
        ly = y2 + th + baseline + pad
    cv2.rectangle(frame, (x1, ly - th - baseline), (x1 + tw + pad, ly + pad//2), color, thickness=-1)
    cv2.putText(frame, label, (x1 + pad//2, ly - baseline//2), font, font_sc, (255, 255, 255), font_th)


def send_alert(frame, filename, caption):
    cv2.imwrite(filename, frame)
    with open(filename, "rb") as photo:
        bot.send_photo(CHAT_ID, photo, caption=caption)


def detect_fire():
    global running, fire_alert_count, smoke_alert_count, last_fire_alert, last_smoke_alert, prev_frame_gray

    while running and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("🎥 Video đã phát hết!")
            break

        now = time.time()
        h, w, _ = frame.shape


        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        motion_detected = False

        if prev_frame_gray is not None:
            frame_delta = cv2.absdiff(prev_frame_gray, gray)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            motion_pixels = cv2.countNonZero(thresh)
            if motion_pixels > MOTION_THRESHOLD:
                motion_detected = True
        else:
            motion_detected = True  

        prev_frame_gray = gray  

        if not motion_detected:
            cv2.imshow("Fire & Smoke Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                running = False
            continue


        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        _, s, v = cv2.split(hsv)

        results = model(frame)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w-1, x2), min(h-1, y2)

                conf = float(box.conf[0])
                cls = int(box.cls[0])


                roi_v = v[y1:y2, x1:x2]
                roi_s = s[y1:y2, x1:x2]
                roi_gray = gray[y1:y2, x1:x2]

                mean_v = roi_v.mean() if roi_v.size > 0 else 0
                mean_s = roi_s.mean() if roi_s.size > 0 else 0
                variance = cv2.Laplacian(roi_gray, cv2.CV_64F).var() if roi_gray.size > 0 else 0

                # ---- FIRE ----
                if cls == 0 and conf > 0.4:
                    draw_label(frame, x1, y1, x2, y2, conf, "Fire", (0, 0, 255))
                    if fire_alert_count < MAX_ALERTS and now - last_fire_alert >= ALERT_INTERVAL:
                        fire_alert_count += 1
                        last_fire_alert = now
                        send_alert(frame, "fire_alert.jpg", "🔥 Cảnh báo cháy!")

                # ---- SMOKE ----
                elif cls == 2 and conf > 0.4:
                    if mean_v < 210 and (mean_s > 50 or variance < 50):
                        draw_label(frame, x1, y1, x2, y2, conf, "Smoke", (255, 0, 0))
                        if smoke_alert_count < MAX_ALERTS and now - last_smoke_alert >= ALERT_INTERVAL:
                            smoke_alert_count += 1
                            last_smoke_alert = now
                            send_alert(frame, "smoke_alert.jpg", "💨 Cảnh báo khói!")

        cv2.imshow("Fire & Smoke Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            running = False
            break

    cap.release()
    cv2.destroyAllWindows()


@bot.message_handler(commands=["stop"])
def stop_detection(message):
    global running
    if str(message.chat.id) == str(CHAT_ID):
        running = False
        bot.send_message(CHAT_ID, "🛑 Đã dừng phát video!")
        print("🛑 Video đã dừng theo lệnh Telegram.")


def start_bot():
    bot.polling()


if __name__ == "__main__":
    threading.Thread(target=start_bot, daemon=True).start()
    detect_fire()
