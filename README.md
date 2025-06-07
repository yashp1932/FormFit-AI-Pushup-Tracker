# FormFit – AI Pushup Tracker 🏋️‍♂️🤖

FormFit is a real-time pushup form tracker that uses computer vision and a custom-trained AI model to detect poor reps and give instant feedback — helping users train smarter and avoid injury.

🎥 [Watch Demo](https://www.youtube.com/watch?v=d_77Bp3EV2Q)

---

## 💡 Why I Built This

Bad pushup form is common and hard to spot without a coach. I wanted to automate form correction using AI and make workouts safer for everyone — from beginners to athletes.

---

## 🧠 How It Works

- Tracks your body through webcam using OpenCV + MediaPipe
- Calculates joint angles (elbow, shoulder, hip)
- Uses a TensorFlow model trained on **1.6M+ data points** to classify form
- Achieves **94.7% accuracy** and gives real-time feedback

---

## 🔧 Tech Stack

Python · OpenCV · MediaPipe · TensorFlow · NumPy  
Prototyped on **Raspberry Pi 4**, scaled to desktop for faster performance.

---

## 🚀 Quickstart

```bash
git clone https://github.com/yashp1932/FormFit-AI-Pushup-Tracker.git
cd FormFit-AI-Pushup-Tracker
pip install -r requirements.txt
python main.py
