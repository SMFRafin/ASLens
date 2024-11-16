import sys
import cv2
import mediapipe as mp
import numpy as np
import pickle
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")

class HandSignDetector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ASLense")
        self.setStyleSheet("""
            QMainWindow {
                background-image: url("bg.png");
                height: full;
                width: full;
                opacity: 75%;
            }
            QPushButton {
                background-color: rgba(37, 99, 235, 0.9);
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 6px;
                font-size: 14px;
                font-weight: 600;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: rgba(29, 78, 216, 0.9);
            }
            QPushButton:pressed {
                background-color: rgba(30, 64, 175, 0.9);
            }
            QPushButton#clearBtn {
                background-color: rgba(220, 38, 38, 0.9);
            }
            QPushButton#clearBtn:hover {
                background-color: rgba(185, 28, 28, 0.9);
            }
            QTextEdit {
                background-color: rgba(38, 38, 38, 0.85);
                border: 2px solid rgba(64, 64, 64, 0.5);
                border-radius: 8px;
                padding: 12px;
                font-size: 18px;
                color: #ffffff;
            }
            QTextEdit:focus {
                border-color: rgba(37, 99, 235, 0.9);
            }
            QWidget#container {
                background-color: rgba(38, 38, 38, 0.85);
                border-radius: 12px;
                padding: 20px;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
            QWidget#videoContainer {
                height: 640px;
                width: 480px;
                border-radius: 12px;
                padding: 16px;
                margin: 8px;
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 12px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
        """)
        # Initialize detection components
        self.init_detection()
        
        # Create the UI
        self.init_ui()
        
        # Setup video timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        self.timer.start(60)

    def init_detection(self):
        model_dict = pickle.load(open('./model.p', 'rb'))
        self.model = model_dict['model']

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(static_image_mode=False, 
                                       min_detection_confidence=0.7,
                                       max_num_hands=1)

        self.labels_dict = {
            0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f',
            6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l',
            12: 'm', 13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r',
            18: 's', 19: 't', 20: 'u', 21: 'v', 22: 'w', 23: 'x',
            24: 'y', 25: 'z', 26: 'space'
        }

        self.last_detected_character = None
        self.fixed_character = ""
        self.delayCounter = 0
        self.start_time = 0
        self.current_text = ""

    def init_ui(self):
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(24)
        layout.setContentsMargins(32, 32, 32, 32)

        # Add title
        title = QLabel("ASLense")
        title.setStyleSheet("""
            font-size: 32px;
            font-weight: bold;
            color: #6305fa;
            padding: 0px 8px;
        """)
        title.setAlignment(Qt.AlignCenter) 
        layout.addWidget(title, 0, Qt.AlignCenter)  
        
        video_container = QWidget()
        video_container.setObjectName("videoContainer")
        video_layout=QVBoxLayout(video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)
        video_layout.setAlignment(Qt.AlignCenter)
        
        self.video_label=QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: rgba(26, 26, 26, 0.85);
                border-radius: 8px;
                padding: 8px;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
        """)
        video_layout.addWidget(self.video_label)
        layout.addWidget(video_container)

        # Add centered buttons between video and text
        buttons_container = QWidget()
        buttons_container.setObjectName("container")
        buttons_container.setStyleSheet("""
            QWidget#container {
                background: transparent;
                border: none;
                padding: 10px;
            }
        """)
        buttons_layout = QHBoxLayout(buttons_container)
        buttons_layout.setSpacing(16)
        buttons_layout.setAlignment(Qt.AlignCenter)
        
        self.toggle_btn = QPushButton("⏸️ Pause")
        self.clear_btn = QPushButton("🗑️ Clear")
        self.clear_btn.setObjectName("clearBtn")
        
        self.toggle_btn.clicked.connect(self.toggle_detection)
        self.clear_btn.clicked.connect(self.clear_text)
        
        buttons_layout.addWidget(self.toggle_btn)
        buttons_layout.addWidget(self.clear_btn)
        layout.addWidget(buttons_container)

        text_container = QWidget()
        text_container.setObjectName("container")
        text_layout = QVBoxLayout(text_container)
        text_layout.setSpacing(16)
        
        # Text header with icon
        text_header = QHBoxLayout()
        text_label = QLabel("✍️ Detected Text")
        text_label.setStyleSheet("""
            font-size: 20px;
            font-weight: bold;
            color: #ffffff;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        """)
        text_header.addWidget(text_label)
        text_header.addStretch()
        text_layout.addLayout(text_header)
        
        self.text_display = QTextEdit()
        self.text_display.setStyleSheet("""
            font-size: 32px;
            line-height: 1.5;
            background-color: rgba(38, 38, 38, 0.75);
        """)
        self.text_display.setReadOnly(True)
        self.text_display.setMinimumHeight(120)
        text_layout.addWidget(self.text_display)
        
        layout.addWidget(text_container)

        # Set window size and position
        self.setMinimumSize(900, 900)
        self.center_window()
    def toggle_detection(self):
        if self.timer.isActive():
            self.timer.stop()
            self.toggle_btn.setText("▶️ Resume")
        else:
            self.timer.start(60)
            self.toggle_btn.setText("⏸️ Pause")

    def center_window(self):
        frame=self.frameGeometry()
        center=QDesktopWidget().availableGeometry().center()
        frame.moveCenter(center)
        self.move(frame.topLeft())

    def clear_text(self):
        self.current_text=""
        self.text_display.clear()
        self.last_detected_character=None
        self.fixed_character=""
        self.delayCounter=0

    def update_text(self, text):
        if text=='space':
            self.current_text+= ' '
        else:
            if not self.current_text: 
                self.current_text=text.capitalize()
            else:
                self.current_text+=text
            
            self.current_text=self.current_text.replace(" i "," I ")
            if self.current_text.startswith("i "):
                self.current_text="I " + self.current_text[2:]
        self.text_display.setText(self.current_text)

    def update_frame(self):
        data_aux = []
        x_=[] 
        y_=[]

        ret,frame=self.cap.read()
        if not ret:
            return

        frame_rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results=self.hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )

                for i in range(len(hand_landmarks.landmark)):
                    x=hand_landmarks.landmark[i].x
                    y=hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x=hand_landmarks.landmark[i].x
                    y=hand_landmarks.landmark[i].y
                    data_aux.append(x-min(x_))
                    data_aux.append(y-min(y_))

                H,W,_=frame.shape
                x1=int(min(x_)*W)-10
                y1=int(min(y_)*H)-10
                x2=int(max(x_)*W)-10
                y2=int(max(y_)*H)-10

                prediction = self.model.predict([np.asarray(data_aux)])
                predicted_character = self.labels_dict[int(prediction[0])]

                # Draw modern-looking bounding box and prediction
                cv2.rectangle(frame,(x1,y1),(x2,y2),(13,13,13),3)  
                cv2.putText(frame, predicted_character.upper(), (x1,y1-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 1.3,(37,99,235),3,
                          cv2.LINE_AA)

                current_time=time.time()
                if predicted_character==self.last_detected_character:
                    if (current_time-self.start_time)>=1.0:
                        self.fixed_character=predicted_character
                        if self.delayCounter==0:
                            self.update_text(self.fixed_character)
                            self.delayCounter=1
                else:
                    self.start_time = current_time
                    self.last_detected_character=predicted_character
                    self.delayCounter=0

        rgb_image=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h,w,ch=rgb_image.shape
        bytes_per_line=ch*w
        qt_image=QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        scaled_image=qt_image.scaled(self.video_label.size(), Qt.KeepAspectRatio)
        self.video_label.setPixmap(QPixmap.fromImage(scaled_image))

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

def main():
    app=QApplication(sys.argv)
    window=HandSignDetector()
    window.show()
    sys.exit(app.exec_())

if __name__=='__main__':
    main()