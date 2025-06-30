# FormFit – AI Pushup Tracker 🏋️‍♂️

## ⚙️ Summary

FormFit is a **real-time AI-powered pushup form tracker** that uses computer vision and a **custom-trained TensorFlow CNN** to detect improper reps and provide instant feedback. The system was prototyped on a Raspberry Pi 4 using MediaPipe for body landmark detection, then scaled to desktop hardware to enable faster, more accurate inference and better scalability.

**Tech Stack:** Python · OpenCV · MediaPipe · TensorFlow · NumPy · Raspberry Pi 4  

**Key Features:** Real-time pushup form analysis · AI-driven accuracy · Custom CNN model · Scalable architecture for multiple users and exercises · Instant feedback for safer training.

---
## 🔍 Explore

📄 [Project Documentation (PDF)](./Project_Documentation.pdf)  
🎥 [Demo Video](https://www.youtube.com/watch?v=d_77Bp3EV2Q)

---

## 💡 Why we Built This

Bad pushup form is common and hard to spot without a coach. We wanted to automate form correction using AI and make workouts safer for everyone, from beginners to athletes.

---

## 🧠 How It Works

FormFit didn’t start as an AI project. It actually started on a **Raspberry Pi 4** with a simple camera, using brute-force math and logic.

- We used **MediaPipe** to extract 3D body landmarks (shoulders, elbows, hips, knees).
- Then calculated joint angles (like elbow flexion, shoulder position, hip alignment) using raw trigonometry.
- Compared those live angles to known “good form” ranges. If the angles were off, the rep was marked as poor.

It worked, but it was **limited**, slow, and didn’t scale well.

So we leveled it up:

- Recorded **over 700 pushups**.
- Extracted **3.6M+ data points** from body landmarks in the videos.
- Trained a **TensorFlow CNN** to learn patterns in joint positions, movement smoothness, and body alignment.
- Achieved **94.7% accuracy** in form classification.
- Switched to desktop for faster inference and real-time feedback.

This transition from raw math to a trained AI model made it:
- **Faster**
- **More accurate**
- **Easier to grow** (can now train on different body types, camera angles, even add new exercises in the future)



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
```

---

## 🔭 Future Ideas for the Project

- 🎙️ **Live Audio Feedback** – Add speaker output during workouts to give real-time advice on form so users can self-correct mid-rep.
- 🌐 **Web App Expansion** – Let anyone upload a pushup video and receive instant form analysis and a detailed breakdown.

---

## 🤝 Connect

If you are interested in learning more about the project, want to help develop it further, or just have questions, feel free to reach out!

📧 yash.panchal1932@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/yashp1932)  

📧 Kalpkansara123@gmail.com (project partner)  
🔗 [LinkedIn](https://www.linkedin.com/in/kalp-kansara123/)  

---
