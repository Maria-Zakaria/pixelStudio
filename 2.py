import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QFileDialog, QStackedWidget,
    QFrame, QGroupBox, QCheckBox, QScrollArea, QSizePolicy,
    QDoubleSpinBox, QSpinBox, QMessageBox, QLineEdit
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QUrl, QMutex, QMutexLocker
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent

# ==========================================
# 0. شريط تقدم قابل للنقر المباشر (Custom Slider)
# ==========================================
class ClickableSlider(QSlider):
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            val = self.minimum() + ((self.maximum() - self.minimum()) * event.x()) / self.width()
            self.setValue(int(val))
            self.sliderMoved.emit(int(val))
            event.accept()
        super().mousePressEvent(event)


# ==========================================
# 1. كلاس معالجة الصور وفلاتر المحاضرات 7، 8، 9 المتقدمة (Image Processor)
# ==========================================
class ImageProcessor:
    @staticmethod
    def adjust_brightness_contrast(img, brightness=0, contrast=0):
        alpha = (contrast + 100) / 100.0
        beta = brightness
        return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    @staticmethod
    def adjust_saturation_hue(img, saturation=0, hue=0):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        if saturation != 0:
            s = cv2.add(s, saturation)
        if hue != 0:
            h = (h.astype(int) + hue) % 180
            h = h.astype(np.uint8)
            
        hsv_mod = cv2.merge([h, s, v])
        return cv2.cvtColor(hsv_mod, cv2.COLOR_HSV2BGR)

    @staticmethod
    def rotate_image(img, angle):
        if angle == 90:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(img, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return img

    @staticmethod
    def flip_image(img, flip_code):
        return cv2.flip(img, flip_code)

    @staticmethod
    def resize_image(img, width, height):
        if width > 0 and height > 0:
            return cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
        return img

    @staticmethod
    def crop_image(img, x, y, w, h):
        ih, iw = img.shape[:2]
        x = max(0, min(x, iw-1))
        y = max(0, min(y, ih-1))
        w = max(1, min(w, iw - x))
        h = max(1, min(h, ih - y))
        return img[y:y+h, x:x+w]

    @staticmethod
    def remove_background_simple(img):
        # إزالة الخلفية باستخدام خوارزمية Otsu Thresholding و GrabCut أو Masking مبسطة
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        
        # تحويل الخلفية إلى شفافة أو بيضاء نقية
        result = img.copy()
        result[sure_bg == 0] = [255, 255, 255]
        return result

    @staticmethod
    def apply_threshold(gray_img, t_value=127):
        _, binary = cv2.threshold(gray_img, t_value, 255, cv2.THRESH_BINARY)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def apply_histogram_equalization(gray_img):
        eq = cv2.equalizeHist(gray_img)
        return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def apply_log_transform(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        c = 255 / np.log(1 + np.max(gray))
        log_img = c * np.log(gray.astype(np.float32) + 1)
        log_img = np.uint8(log_img)
        return cv2.cvtColor(log_img, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def apply_gamma_transform(img, gamma=1.5):
        norm = img.astype(np.float32) / 255.0
        gamma_img = np.power(norm, gamma)
        return np.uint8(gamma_img * 255)

    @staticmethod
    def filter_grayscale(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def filter_sepia(img):
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        sepia = cv2.transform(img, kernel)
        return np.clip(sepia, 0, 255).astype(np.uint8)

    @staticmethod
    def filter_blur_gaussian(img, ksize=15):
        if ksize % 2 == 0: ksize += 1
        return cv2.GaussianBlur(img, (ksize, ksize), 0)

    # --- فلاتر المحاضرات (Laplacian, Sobel, Canny, Morphology, Watershed) ---
    @staticmethod
    def filter_laplacian_sharpen(img):
        # Laplacian-Based Sharpening (محاضرة 7)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        return cv2.filter2D(img, -1, kernel)

    @staticmethod
    def filter_sobel_edges(img):
        # Sobel Edge Detection (محاضرة 7)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_x = cv2.convertScaleAbs(sobel_x)
        sobel_y = cv2.convertScaleAbs(sobel_y)
        sobel_xy = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
        return cv2.cvtColor(sobel_xy, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def filter_canny_edges(img):
        # Canny Edge Detection (محاضرة 7)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 100, 200)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def filter_morphology_gradient(img):
        # Morphological Gradient (محاضرة 9)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((5, 5), np.uint8)
        gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        return cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def filter_sketch(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        inv = cv2.bitwise_not(gray)
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        sketch = cv2.divide(gray, 255 - blur, scale=256)
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def filter_warm(img):
        b, g, r = cv2.split(img)
        r = cv2.add(r, 30)
        b = cv2.subtract(b, 20)
        return cv2.merge([b, g, r])

    @staticmethod
    def filter_cool(img):
        b, g, r = cv2.split(img)
        b = cv2.add(b, 30)
        r = cv2.subtract(r, 20)
        return cv2.merge([b, g, r])

    @staticmethod
    def filter_cinematic(img):
        img_temp = ImageProcessor.adjust_brightness_contrast(img, brightness=-10, contrast=20)
        b, g, r = cv2.split(img_temp)
        b = cv2.add(b, 15)
        r = cv2.add(r, 10)
        return cv2.merge([b, g, r])


# ==========================================
# 2. كلاس كشف الوجوه والتعديل التجميلي والفلاتر المتقدمة (Face Detector & Makeup)
# ==========================================
class FaceDetector:
    def __init__(self):
        self.face_cascade = None
        try:
            if hasattr(cv2, 'CascadeClassifier'):
                cascade_path = getattr(cv2, 'data', None)
                if cascade_path and hasattr(cascade_path, 'haarcascades'):
                    xml_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                else:
                    xml_file = 'haarcascade_frontalface_default.xml'
                self.face_cascade = cv2.CascadeClassifier(xml_file)
        except Exception:
            self.face_cascade = None

    def detect_and_process(self, frame, mode="الأصلي"):
        if self.face_cascade is None or self.face_cascade.empty():
            return frame
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            for (x, y, w, h) in faces:
                if mode == "تحديد الوجه":
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, "Detected Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                elif mode == "تجميل وميكب":
                    # تطبيق تأثير تنعيم وتوريد خفيف على منطقة الوجه (Makeup & Smoothing)
                    face_roi = frame[y:y+h, x:x+w]
                    smooth = cv2.bilateralFilter(face_roi, 9, 75, 75)
                    # إضافة لمسة لونية وردية شفافة
                    pink_overlay = np.full(face_roi.shape, (200, 180, 255), dtype=np.uint8)
                    blended = cv2.addWeighted(smooth, 0.8, pink_overlay, 0.2, 0)
                    frame[y:y+h, x:x+w] = blended
                    cv2.putText(frame, "Smart Makeup Filter", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

                elif mode == "تكبير الفم":
                    # تحديد منطقة النصف السفلي من الوجه لتكبيرها (تأثير بصري تفاعلي)
                    mouth_roi = frame[y + int(h*0.6):y+h, x + int(w*0.2):x + int(w*0.8)]
                    if mouth_roi.shape[0] > 10 and mouth_roi.shape[1] > 10:
                        resized_mouth = cv2.resize(mouth_roi, (int(mouth_roi.shape[1]*1.2), int(mouth_roi.shape[0]*1.2)))
                        # اقتصاب أو دمج لمنع خروج الأبعاد
                        mh, mw = mouth_roi.shape[:2]
                        rmh, rmw = resized_mouth.shape[:2]
                        if rmh >= mh and rmw >= mw:
                            cropped_res = resized_mouth[:mh, :mw]
                            frame[y + int(h*0.6):y + int(h*0.6) + mh, x + int(w*0.2):x + int(w*0.2) + mw] = cropped_res
                    cv2.putText(frame, "Mouth Zoom Filter", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

                elif mode == "كشف الحواف Canny":
                    face_roi = frame[y:y+h, x:x+w]
                    gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(gray_face, 100, 200)
                    frame[y:y+h, x:x+w] = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                    cv2.putText(frame, "Canny Face Edges", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                elif mode == "مرشح لابلاسيان":
                    face_roi = frame[y:y+h, x:x+w]
                    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
                    sharp_face = cv2.filter2D(face_roi, -1, kernel)
                    frame[y:y+h, x:x+w] = sharp_face
                    cv2.putText(frame, "Laplacian Face Sharp", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        except Exception:
            pass
        return frame


# ==========================================
# 3. خيط معالجة الفيديو الآمن (Video Thread)
# ==========================================
class VideoThread(QThread):
    change_frame_signal = pyqtSignal(np.ndarray)
    position_signal = pyqtSignal(int, int)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self._run_flag = True
        self.is_paused = False
        self.current_filter = "الأصلي"
        self.speed = 1.0
        self.mutex = QMutex()
        
        self.cap = cv2.VideoCapture(self.video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30

    def run(self):
        while self._run_flag:
            if self.is_paused:
                self.msleep(50)
                continue

            with QMutexLocker(self.mutex):
                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if not ret:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue

                    current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                    self.position_signal.emit(current_frame, self.total_frames)
                else:
                    break

            frame = self.process_frame(frame)
            self.change_frame_signal.emit(frame)
            
            delay = int((1000 / self.fps) / self.speed)
            self.msleep(max(1, delay))

    def process_frame(self, frame):
        if self.current_filter == "أبيض وأسود":
            return ImageProcessor.filter_grayscale(frame)
        elif self.current_filter == "سيبيا":
            return ImageProcessor.filter_sepia(frame)
        elif self.current_filter == "دافئ":
            return ImageProcessor.filter_warm(frame)
        elif self.current_filter == "بارد":
            return ImageProcessor.filter_cool(frame)
        elif self.current_filter == "تمويه":
            return ImageProcessor.filter_blur_gaussian(frame)
        elif self.current_filter == "لابلاسيان حاد":
            return ImageProcessor.filter_laplacian_sharpen(frame)
        elif self.current_filter == "حواف صوبل":
            return ImageProcessor.filter_sobel_edges(frame)
        elif self.current_filter == "حواف كاني":
            return ImageProcessor.filter_canny_edges(frame)
        elif self.current_filter == "تدرج مورفولوجي":
            return ImageProcessor.filter_morphology_gradient(frame)
        elif self.current_filter == "رسم":
            return ImageProcessor.filter_sketch(frame)
        return frame

    def pause(self):
        self.is_paused = True

    def play(self):
        self.is_paused = False

    def stop_playback(self):
        self.is_paused = True
        with QMutexLocker(self.mutex):
            if self.cap and self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def seek(self, frame_number):
        with QMutexLocker(self.mutex):
            if self.cap and self.cap.isOpened():
                frame_number = max(0, min(frame_number, self.total_frames - 1))
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    def set_speed(self, speed):
        self.speed = max(0.25, min(4.0, speed))

    def stop(self):
        self._run_flag = False
        self.wait()
        with QMutexLocker(self.mutex):
            if self.cap:
                self.cap.release()


# ==========================================
# 4. الواجهة الرئيسية والتطبيق (Main Window)
# ==========================================
class PixelStudioApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PixelStudio - Vision & Image Processing Suite")
        self.setGeometry(50, 50, 1350, 850)
        self.setLayoutDirection(Qt.RightToLeft)

        self.original_image = None
        self.processed_image = None
        self.current_filter = "الأصلي"

        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.update_camera_frame)
        self.cap = None
        self.face_detector = FaceDetector()
        self.camera_mode = "الأصلي"

        self.current_video_path = None

        self.init_ui()
        self.apply_stylesheet()

    def init_ui(self):
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # القائمة الجانبية
        sidebar = QFrame()
        sidebar.setFixedWidth(230)
        sidebar.setObjectName("Sidebar")
        sidebar_layout = QVBoxLayout(sidebar)
        
        logo = QLabel("PixelStudio 🚀")
        logo.setFont(QFont("Segoe UI", 16, QFont.Bold))
        logo.setStyleSheet("color: #38BDF8; margin-bottom: 20px;")
        logo.setAlignment(Qt.AlignCenter)
        sidebar_layout.addWidget(logo)

        btn_home = QPushButton("🏠 الرئيسية")
        btn_image = QPushButton("🖼️ تحرير الصور")
        btn_video = QPushButton("🎬 تحرير الفيديو")
        btn_camera = QPushButton("📷 الكاميرا الذكية")
        btn_about = QPushButton("ℹ️ عن البرنامج والمشروع")

        for btn in [btn_home, btn_image, btn_video, btn_camera, btn_about]:
            btn.setFixedHeight(45)
            btn.setCursor(Qt.PointingHandCursor)
            sidebar_layout.addWidget(btn)

        btn_home.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))
        btn_image.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(1))
        btn_video.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(2))
        btn_camera.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(3))
        btn_about.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(4))

        sidebar_layout.addStretch()

        # تكديس الواجهات (Stack)
        self.stacked_widget = QStackedWidget()
        self.stacked_widget.addWidget(self.create_home_view())
        self.stacked_widget.addWidget(self.create_image_editor_view())
        self.stacked_widget.addWidget(self.create_video_editor_view())
        self.stacked_widget.addWidget(self.create_camera_view())
        self.stacked_widget.addWidget(self.create_about_view())

        main_layout.addWidget(sidebar)
        main_layout.addWidget(self.stacked_widget)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    # ==========================================
    # واجهة الصفحة الرئيسية (الجديدة كلياً بكلام مؤثر)
    # ==========================================
    def create_home_view(self):
        page = QWidget()
        page.setStyleSheet("""
            QWidget { background-color: #0F172A; }
            #HeroTitle {
                color: #F8FAFC;
                font-size: 26px;
                font-weight: 800;
                font-family: 'Segoe UI', 'Cairo', sans-serif;
            }
            #InfoBox {
                background-color: #1E293B;
                border: 1px solid #334155;
                border-radius: 14px;
                padding: 20px;
            }
            #HeroBadge {
                color: #38BDF8;
                font-size: 15px;
                font-weight: bold;
            }
            #HeroSubtitle {
                color: #94A3B8;
                font-size: 14px;
                line-height: 1.6;
            }
            #FeatureCard {
                background-color: #1E293B;
                border: 1px solid #334155;
                border-radius: 16px;
                padding: 22px;
            }
            #FeatureCard:hover {
                border-color: #38BDF8;
            }
            #CardTitle {
                color: #F1F5F9;
                font-size: 17px;
                font-weight: bold;
            }
            #CardDesc {
                color: #94A3B8;
                font-size: 13px;
            }
            #ActionBtn {
                background-color: #0284C7;
                color: white;
                border-radius: 8px;
                padding: 10px 16px;
                font-size: 13px;
                font-weight: bold;
                border: none;
            }
            #ActionBtn:hover {
                background-color: #0369A1;
            }
        """)

        main_layout = QVBoxLayout(page)
        main_layout.setContentsMargins(40, 30, 40, 30)
        main_layout.setSpacing(22)

        title = QLabel("✨ لوحة القيادة المركزية لنظام معالجة ورؤية الحواسيب")
        title.setObjectName("HeroTitle")

        info_box = QFrame()
        info_box.setObjectName("InfoBox")
        info_layout = QVBoxLayout(info_box)
        info_layout.setSpacing(10)

        badge = QLabel(" مشروع تخرج / مقرر معالجة الصور الرقمية المتقدمة بإشراف الأستاذة المميزة: م/ سمية الشيبة")
        badge.setObjectName("HeroBadge")

        subtitle = QLabel(
            "هذا النظام الثوري وُلد ليكون ثمرة جهد هندسي متكامل ومثابرة علمية وفنية عالية. "
            "يمثل هذا العمل إبداعاً تقنياً تم تطويره بكل شغف وحرفية لربط النظريات الأكاديمية العميقة بالتطبيقات العملية الملموسة. "
            "نقدم لكم نافذة ذكية تدمج الخوارزميات الرياضية الدقيقة مع واجهات عصرية تلبي طموحات مهندسي المستقبل."
        )
        subtitle.setObjectName("HeroSubtitle")
        subtitle.setWordWrap(True)

        info_layout.addWidget(badge)
        info_layout.addWidget(subtitle)

        # بطاقات الخدمات السريعة
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(20)

        features = [
            ("معالجة وتحرير الصور", "تعديل هندسي شامل، قص، تغيير الحجم، إزالة الخلفية، وتطبيق مرشحات الفضاء المكاني.", 1),
            ("معالجة وتحليل الفيديو", "التحكم بالزمن، استخراج الفريمات، قص وتقسيم المقاطع، وتطبيق خوارزميات الفلاتر الحية.", 2),
            ("رؤية الكاميرا الذكية", "اكتشاف الوجوه، تجميل وميكب ذكي، تكبير الفم التفاعلي، وفلاتر الكاميرا المتقدمة المستوحاة من المحاضرات.", 3)
        ]

        for card_title, card_desc, page_idx in features:
            card = QFrame()
            card.setObjectName("FeatureCard")
            c_layout = QVBoxLayout(card)
            c_layout.setSpacing(12)

            lbl_title = QLabel(card_title)
            lbl_title.setObjectName("CardTitle")

            lbl_desc = QLabel(card_desc)
            lbl_desc.setObjectName("CardDesc")
            lbl_desc.setWordWrap(True)

            btn = QPushButton("استكشف الأقسام ←")
            btn.setObjectName("ActionBtn")
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda checked, idx=page_idx: self.stacked_widget.setCurrentIndex(idx))

            c_layout.addWidget(lbl_title)
            c_layout.addWidget(lbl_desc)
            c_layout.addStretch()
            c_layout.addWidget(btn)

            cards_layout.addWidget(card, 1)

        main_layout.addWidget(title)
        main_layout.addWidget(info_box)
        main_layout.addLayout(cards_layout)
        main_layout.addStretch()

        return page

    # ==========================================
    # واجهة تحرير الصور (محدثة مع قص، تغيير الحجم، إزالة الخلفية، وفلاتر المحاضرات)
    # ==========================================
    def create_image_editor_view(self):
        page = QWidget()
        page.setStyleSheet("background-color: #0F172A; color: #F8FAFC;")
        layout = QHBoxLayout(page)

        center_panel = QVBoxLayout()
        top_bar = QHBoxLayout()
        
        btn_open = QPushButton("📂 فتح صورة")
        btn_open.clicked.connect(self.open_image)
        btn_save = QPushButton("💾 حفظ الصورة")
        btn_save.clicked.connect(self.save_image)
        btn_reset = QPushButton("🔄 إعادة تعيين")
        btn_reset.clicked.connect(self.reset_image_controls)

        top_bar.addWidget(btn_open)
        top_bar.addWidget(btn_save)
        top_bar.addWidget(btn_reset)
        top_bar.addStretch()

        self.img_display_label = QLabel("قم بفتح صورة للبدء في عمليات المعالجة المتقدمة")
        self.img_display_label.setAlignment(Qt.AlignCenter)
        self.img_display_label.setStyleSheet("border: 2px dashed #334155; background-color: #1E293B; border-radius: 10px;")
        self.img_display_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        center_panel.addLayout(top_bar)
        center_panel.addWidget(self.img_display_label, 1)

        # شريط الفلاتر السفلية المحدثة (تتضمن فلاتر المحاضرات 7، 8، 9)
        filters_scroll = QScrollArea()
        filters_scroll.setFixedHeight(95)
        filters_widget = QWidget()
        filters_layout = QHBoxLayout(filters_widget)

        filter_list = [
            "الأصلي", "أبيض وأسود", "سيبيا", "دافئ", "بارد", "سينمائي", "تمويه", 
            "لابلاسيان حاد (محاضرة 7)", "حواف صوبل (محاضرة 7)", "حواف كاني (محاضرة 7)", 
            "تدرج مورفولوجي (محاضرة 9)", "رسم", "تعديل الهستغرام", "تحويل لوغاريتمي"
        ]
        for f in filter_list:
            btn = QPushButton(f)
            btn.clicked.connect(lambda checked, name=f: self.apply_image_filter(name))
            filters_layout.addWidget(btn)

        filters_scroll.setWidget(filters_widget)
        filters_scroll.setWidgetResizable(True)
        center_panel.addWidget(filters_scroll)

        # لوحة التحكم الجانبية للصور (تتضمن القص، تغيير الحجم، وإزالة الخلفية)
        control_panel = QScrollArea()
        control_panel.setFixedWidth(310)
        control_widget = QWidget()
        controls_layout = QVBoxLayout(control_widget)

        controls_layout.addWidget(QLabel("🛠️ أدوات ومعالجة الصور المتقدمة", font=QFont("Segoe UI", 11, QFont.Bold)))

        # مجموعة الهندسة والقص وإزالة الخلفية
        geom_group = QGroupBox("العمليات الهندسية والتحويلات")
        geom_layout = QVBoxLayout(geom_group)
        
        btn_rot = QPushButton("🔄 تدوير 90 درجة")
        btn_rot.clicked.connect(lambda: self.transform_image("rotate"))
        btn_flip_h = QPushButton("↔️ قلب أفقي")
        btn_flip_h.clicked.connect(lambda: self.transform_image("flip_h"))
        btn_flip_v = QPushButton("↕️ قلب عمودي")
        btn_flip_v.clicked.connect(lambda: self.transform_image("flip_v"))
        btn_bg_rem = QPushButton("✂️ إزالة الخلفية التلقائية")
        btn_bg_rem.clicked.connect(self.remove_background_action)

        geom_layout.addWidget(btn_rot)
        geom_layout.addWidget(btn_flip_h)
        geom_layout.addWidget(btn_flip_v)
        geom_layout.addWidget(btn_bg_rem)
        controls_layout.addWidget(geom_group)

        # مجموعة تغيير الحجم (Resize)
        resize_group = QGroupBox("تغيير الحجم (Resize)")
        resize_layout = QVBoxLayout(resize_group)
        
        h_w = QHBoxLayout()
        h_w.addWidget(QLabel("العرض:"))
        self.spin_w = QSpinBox()
        self.spin_w.setRange(10, 4000)
        self.spin_w.setValue(800)
        h_w.addWidget(self.spin_w)

        h_h = QHBoxLayout()
        h_h.addWidget(QLabel("الطول:"))
        self.spin_h = QSpinBox()
        self.spin_h.setRange(10, 4000)
        self.spin_h.setValue(600)
        h_h.addWidget(self.spin_h)

        btn_resize = QPushButton("تطبيق تغيير الحجم")
        btn_resize.clicked.connect(self.resize_image_action)

        resize_layout.addLayout(h_w)
        resize_layout.addLayout(h_h)
        resize_layout.addWidget(btn_resize)
        controls_layout.addWidget(resize_group)

        # مجموعة القص (Crop)
        crop_group = QGroupBox("قص الصورة (Crop)")
        crop_layout = QVBoxLayout(crop_group)
        
        grid_crop = QHBoxLayout()
        self.spin_cx = QSpinBox(); self.spin_cx.setRange(0, 2000); self.spin_cx.setValue(50)
        self.spin_cy = QSpinBox(); self.spin_cy.setRange(0, 2000); self.spin_cy.setValue(50)
        self.spin_cw = QSpinBox(); self.spin_cw.setRange(10, 3000); self.spin_cw.setValue(400)
        self.spin_ch = QSpinBox(); self.spin_ch.setRange(10, 3000); self.spin_ch.setValue(300)

        grid_crop.addWidget(QLabel("X:"))
        grid_crop.addWidget(self.spin_cx)
        grid_crop.addWidget(QLabel("Y:"))
        grid_crop.addWidget(self.spin_cy)
        
        grid_crop2 = QHBoxLayout()
        grid_crop2.addWidget(QLabel("W:"))
        grid_crop2.addWidget(self.spin_cw)
        grid_crop2.addWidget(QLabel("H:"))
        grid_crop2.addWidget(self.spin_ch)

        btn_crop_act = QPushButton("تنفيذ القص")
        btn_crop_act.clicked.connect(self.crop_image_action)

        crop_layout.addLayout(grid_crop)
        crop_layout.addLayout(grid_crop2)
        crop_layout.addWidget(btn_crop_act)
        controls_layout.addWidget(crop_group)

        # الضبط والتحسين
        adj_group = QGroupBox("الضبط والتحسين (Brightness/Gamma)")
        adj_layout = QVBoxLayout(adj_group)

        adj_layout.addWidget(QLabel("السطوع:"))
        self.slider_bright = QSlider(Qt.Horizontal)
        self.slider_bright.setRange(-100, 100); self.slider_bright.setValue(0)
        self.slider_bright.valueChanged.connect(self.update_image_display)
        adj_layout.addWidget(self.slider_bright)

        adj_layout.addWidget(QLabel("التباين:"))
        self.slider_contrast = QSlider(Qt.Horizontal)
        self.slider_contrast.setRange(-100, 100); self.slider_contrast.setValue(0)
        self.slider_contrast.valueChanged.connect(self.update_image_display)
        adj_layout.addWidget(self.slider_contrast)

        adj_layout.addWidget(QLabel("تصحيح جاما:"))
        self.slider_gamma = QSlider(Qt.Horizontal)
        self.slider_gamma.setRange(1, 30); self.slider_gamma.setValue(10)
        self.slider_gamma.valueChanged.connect(self.update_image_display)
        adj_layout.addWidget(self.slider_gamma)

        controls_layout.addWidget(adj_group)
        control_panel.setWidget(control_widget)
        control_panel.setWidgetResizable(True)

        layout.addLayout(center_panel, 1)
        layout.addWidget(control_panel)
        return page

    # ==========================================
    # واجهة تحرير الفيديو
    # ==========================================
    def create_video_editor_view(self):
        page = QWidget()
        page.setStyleSheet("background-color: #0F172A; color: #F8FAFC;")
        layout = QHBoxLayout(page)

        v_panel = QVBoxLayout()
        top_v = QHBoxLayout()
        
        btn_open_v = QPushButton("🎬 فتح فيديو")
        btn_open_v.clicked.connect(self.open_video)
        top_v.addWidget(btn_open_v)
        top_v.addStretch()

        self.video_display_label = QLabel("قم بتشغيل مقطع فيديو للبدء في المعالجة الحية")
        self.video_display_label.setAlignment(Qt.AlignCenter)
        self.video_display_label.setStyleSheet("border: 2px dashed #334155; background-color: #000000; border-radius: 10px;")
        self.video_display_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        self.video_slider = ClickableSlider(Qt.Horizontal)
        self.video_slider.setRange(0, 1000)
        self.video_slider.sliderMoved.connect(self.set_video_position)

        player_ctrl_layout = QHBoxLayout()
        btn_play = QPushButton("▶️ تشغيل")
        btn_pause = QPushButton("⏸️ إيقاف مؤقت")
        btn_stop = QPushButton("⏹️ إيقاف")
        btn_rewind = QPushButton("⏪ رجوع (5s)")
        btn_forward = QPushButton("⏩ تقديم (5s)")

        btn_play.clicked.connect(self.play_video)
        btn_pause.clicked.connect(self.pause_video)
        btn_stop.clicked.connect(self.stop_video)
        btn_rewind.clicked.connect(lambda: self.seek_relative(-5))
        btn_forward.clicked.connect(lambda: self.seek_relative(5))

        player_ctrl_layout.addWidget(btn_play)
        player_ctrl_layout.addWidget(btn_pause)
        player_ctrl_layout.addWidget(btn_stop)
        player_ctrl_layout.addWidget(btn_rewind)
        player_ctrl_layout.addWidget(btn_forward)

        v_panel.addLayout(top_v)
        v_panel.addWidget(self.video_display_label, 1)
        v_panel.addWidget(self.video_slider)
        v_panel.addLayout(player_ctrl_layout)

        v_controls_scroll = QScrollArea()
        v_controls_scroll.setFixedWidth(310)
        v_controls_widget = QWidget()
        v_ctrl_layout = QVBoxLayout(v_controls_widget)

        speed_group = QGroupBox("سرعة الفيديو")
        speed_layout = QHBoxLayout(speed_group)
        speed_layout.addWidget(QLabel("السرعة:"))
        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(0.25, 4.0); self.speed_spin.setSingleStep(0.25); self.speed_spin.setValue(1.0)
        self.speed_spin.valueChanged.connect(self.change_video_speed)
        speed_layout.addWidget(self.speed_spin)
        v_ctrl_layout.addWidget(speed_group)

        filter_group = QGroupBox("فلاتر الفيديو المباشرة")
        filter_layout = QVBoxLayout(filter_group)
        v_filters = [
            "الأصلي", "أبيض وأسود", "سيبيا", "دافئ", "بارد", "تمويه", 
            "لابلاسيان حاد", "حواف صوبل", "حواف كاني", "تدرج مورفولوجي", "رسم"
        ]
        for vf in v_filters:
            btn = QPushButton(vf)
            btn.clicked.connect(lambda checked, name=vf: self.apply_video_filter(name))
            filter_layout.addWidget(btn)
        v_ctrl_layout.addWidget(filter_group)

        v_ctrl_layout.addStretch()
        v_controls_scroll.setWidget(v_controls_widget)
        v_controls_scroll.setWidgetResizable(True)

        layout.addLayout(v_panel, 1)
        layout.addWidget(v_controls_scroll)
        return page

    # ==========================================
    # واجهة الكاميرا الذكية (تتضمن فلاتر الوجوه الـ 5 المطلوبة: تحديد الوجه، ميكب، تكبير الفم، كاني، لابلاسيان)
    # ==========================================
    def create_camera_view(self):
        page = QWidget()
        page.setStyleSheet("background-color: #0F172A; color: #F8FAFC;")
        layout = QHBoxLayout(page)

        cam_panel = QVBoxLayout()
        self.cam_display_label = QLabel("اضغط 'تشغيل الكاميرا' لتفعيل فلاتر الرؤية والذكاء الاصطناعي الحية")
        self.cam_display_label.setAlignment(Qt.AlignCenter)
        self.cam_display_label.setStyleSheet("border: 2px dashed #334155; background-color: #000000; border-radius: 10px;")
        self.cam_display_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        cam_btn_layout = QHBoxLayout()
        btn_start_cam = QPushButton("🟢 تشغيل الكاميرا")
        btn_start_cam.clicked.connect(self.start_camera)
        btn_stop_cam = QPushButton("🔴 إيقاف الكاميرا")
        btn_stop_cam.clicked.connect(self.stop_camera)
        btn_capture = QPushButton("📸 التقاط وحفظ الصورة")
        btn_capture.clicked.connect(self.capture_photo)

        cam_btn_layout.addWidget(btn_start_cam)
        cam_btn_layout.addWidget(btn_stop_cam)
        cam_btn_layout.addWidget(btn_capture)

        cam_panel.addLayout(cam_btn_layout)
        cam_panel.addWidget(self.cam_display_label, 1)

        cam_controls = QScrollArea()
        cam_controls.setFixedWidth(310)
        cam_ctrl_widget = QWidget()
        cam_ctrl_layout = QVBoxLayout(cam_ctrl_widget)

        cam_ctrl_layout.addWidget(QLabel("✨ فلاتر الكاميرا المتقدمة والمطلوبة", font=QFont("Segoe UI", 11, QFont.Bold)))
        cam_ctrl_layout.addWidget(QLabel("أكثر من 5 فلاتر تفاعلية تعتمد على التحديد الذكي للوجوه والحواف:"))

        camera_modes = [
            "الأصلي",
            "تحديد الوجه",
            "تجميل وميكب",
            "تكبير الفم",
            "كشف الحواف Canny",
            "مرشح لابلاسيان"
        ]

        for mode in camera_modes:
            btn = QPushButton(mode)
            btn.setFixedHeight(40)
            btn.clicked.connect(lambda checked, m=mode: self.set_camera_mode(m))
            cam_ctrl_layout.addWidget(btn)

        cam_ctrl_layout.addStretch()
        cam_controls.setWidget(cam_ctrl_widget)
        cam_controls.setWidgetResizable(True)

        layout.addLayout(cam_panel, 1)
        layout.addWidget(cam_controls)
        return page

    # ==========================================
    # واجهة "عن البرنامج والمشروع" (معلومات كاملة ونص مؤثر)
    # ==========================================
    def create_about_view(self):
        page = QWidget()
        page.setStyleSheet("background-color: #0F172A; color: #F8FAFC;")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(50, 40, 50, 40)
        layout.setSpacing(20)

        title = QLabel("ℹ️ حول نظام PixelStudio وبرمجيات الرؤية الحاسوبية")
        title.setFont(QFont("Segoe UI", 20, QFont.Bold))
        title.setStyleSheet("color: #38BDF8;")

        content_box = QFrame()
        content_box.setStyleSheet("background-color: #1E293B; border: 1px solid #334155; border-radius: 14px; padding: 30px;")
        box_layout = QVBoxLayout(content_box)
        box_layout.setSpacing(15)

        lbl_desc = QLabel(
            "🌟 قصة نجاح وإبداع أكاديمي:\n\n"
            "تم تصميم وبرمجة هذا النظام المتكامل كجزء من متطلبات مقرر معالجة الصور الرقمية المتقدمة، "
            "تحت الإشراف المباشر والتوجيه العلمي السديد من الأستاذة القديرة: (م/ سمية الشيبة).\n\n"
            "هذا المشروع هو ثمرة تعاون وتكاتف هندسي رفيع المستوى بين نخبة من المهندسات المبدعات:\n"
            "• براءة القاضي\n"
            "• حسناء المغربي\n"
            "• ماريا الورافي\n"
            "🎯 أهداف النظام:\n"
            "دمج الخوارزميات الرياضية المعقدة الخاصة بالمجال المكاني وتحويلات فورييه، مرشحات لابلاسيان، كشف الحواف (Sobel & Canny)، "
            "العمليات المورفولوجية المتقدمة، وخوارزميات التجزئة Segmentation وتحويلات Hough، ضمن واجهة تفاعلية احترافية وسهلة الاستخدام.\n\n"
            "نأمل أن يكون هذا العمل مرجعاً حقيقياً ونموذجاً متميزاً يجمع بين روعة الفن ودقة الهندسة."
        )
        lbl_desc.setFont(QFont("Segoe UI", 13))
        lbl_desc.setWordWrap(True)
        lbl_desc.setStyleSheet("color: #CBD5E1; line-height: 1.6;")

        box_layout.addWidget(lbl_desc)
        layout.addWidget(title)
        layout.addWidget(content_box)
        layout.addStretch()

        return page

    # ==========================================
    # الوظائف والأحداث (Image & Video & Camera Actions)
    # ==========================================
    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "اختر صورة", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)")
        if path:
            self.original_image = cv2.imread(path)
            self.processed_image = self.original_image.copy()
            self.update_image_display()

    def save_image(self):
        if self.processed_image is not None:
            path, _ = QFileDialog.getSaveFileName(self, "حفظ الصورة", "", "PNG (*.png);;JPEG (*.jpg)")
            if path:
                cv2.imwrite(path, self.processed_image)
                QMessageBox.information(self, "نجاح", "تم حفظ الصورة بنجاح!")

    def update_image_display(self):
        if self.original_image is None:
            return

        img = self.original_image.copy()

        bright = self.slider_bright.value()
        contrast = self.slider_contrast.value()
        if bright != 0 or contrast != 0:
            img = ImageProcessor.adjust_brightness_contrast(img, bright, contrast)

        gamma_val = self.slider_gamma.value() / 10.0
        if gamma_val != 1.0:
            img = ImageProcessor.apply_gamma_transform(img, gamma=gamma_val)

        if self.current_filter == "أبيض وأسود":
            img = ImageProcessor.filter_grayscale(img)
        elif self.current_filter == "سيبيا":
            img = ImageProcessor.filter_sepia(img)
        elif self.current_filter == "دافئ":
            img = ImageProcessor.filter_warm(img)
        elif self.current_filter == "بارد":
            img = ImageProcessor.filter_cool(img)
        elif self.current_filter == "سينمائي":
            img = ImageProcessor.filter_cinematic(img)
        elif self.current_filter == "تمويه":
            img = ImageProcessor.filter_blur_gaussian(img)
        elif self.current_filter == "لابلاسيان حاد":
            img = ImageProcessor.filter_laplacian_sharpen(img)
        elif self.current_filter == "حواف صوبل ":
            img = ImageProcessor.filter_sobel_edges(img)
        elif self.current_filter == "حواف كاني ":
            img = ImageProcessor.filter_canny_edges(img)
        elif self.current_filter == "تدرج مورفولوجي ":
            img = ImageProcessor.filter_morphology_gradient(img)
        elif self.current_filter == "رسم":
            img = ImageProcessor.filter_sketch(img)
        elif self.current_filter == "تعديل الهستغرام":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = ImageProcessor.apply_histogram_equalization(gray)
        elif self.current_filter == "تحويل لوغاريتمي":
            img = ImageProcessor.apply_log_transform(img)

        self.processed_image = img
        self.display_cv_image(self.img_display_label, img)

    def apply_image_filter(self, filter_name):
        self.current_filter = filter_name
        self.update_image_display()

    def transform_image(self, action):
        if self.original_image is None: return

        if action == "rotate":
            self.original_image = ImageProcessor.rotate_image(self.original_image, 90)
        elif action == "flip_h":
            self.original_image = ImageProcessor.flip_image(self.original_image, 1)
        elif action == "flip_v":
            self.original_image = ImageProcessor.flip_image(self.original_image, 0)

        self.update_image_display()

    def resize_image_action(self):
        if self.original_image is None: return
        w = self.spin_w.value()
        h = self.spin_h.value()
        self.original_image = ImageProcessor.resize_image(self.original_image, w, h)
        self.update_image_display()
        QMessageBox.information(self, "تغيير الحجم", f"تم تغيير أبعاد الصورة إلى {w}x{h}")

    def crop_image_action(self):
        if self.original_image is None: return
        x = self.spin_cx.value()
        y = self.spin_cy.value()
        w = self.spin_cw.value()
        h = self.spin_ch.value()
        self.original_image = ImageProcessor.crop_image(self.original_image, x, y, w, h)
        self.update_image_display()
        QMessageBox.information(self, "قص الصورة", "تم قص الجزء المحدد بنجاح!")

    def remove_background_action(self):
        if self.original_image is None: return
        self.original_image = ImageProcessor.remove_background_simple(self.original_image)
        self.update_image_display()
        QMessageBox.information(self, "إزالة الخلفية", "تم تطبيق إزالة الخلفية بناءً على خوارزميات التجزئة (Segmentation)!")

    def reset_image_controls(self):
        self.slider_bright.setValue(0)
        self.slider_contrast.setValue(0)
        self.slider_gamma.setValue(10)
        self.current_filter = "الأصلي"
        self.update_image_display()

    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "اختر فيديو", "", "Video Files (*.mp4 *.avi *.mkv)")
        if path:
            self.current_video_path = path
            if hasattr(self, 'video_thread') and self.video_thread.isRunning():
                self.video_thread.stop()

            self.video_thread = VideoThread(path)
            self.video_thread.change_frame_signal.connect(lambda frame: self.display_cv_image(self.video_display_label, frame))
            self.video_thread.position_signal.connect(self.update_video_slider)
            self.video_thread.start()

    def play_video(self):
        if hasattr(self, 'video_thread'): self.video_thread.play()

    def pause_video(self):
        if hasattr(self, 'video_thread'): self.video_thread.pause()

    def stop_video(self):
        if hasattr(self, 'video_thread'): self.video_thread.stop_playback()

    def seek_relative(self, seconds):
        if hasattr(self, 'video_thread') and self.video_thread.cap:
            current_frame = int(self.video_thread.cap.get(cv2.CAP_PROP_POS_FRAMES))
            target_frame = current_frame + int(seconds * self.video_thread.fps)
            self.video_thread.seek(target_frame)

    def set_video_position(self, value):
        if hasattr(self, 'video_thread') and self.video_thread.total_frames > 0:
            target_frame = int((value / 1000.0) * self.video_thread.total_frames)
            self.video_thread.seek(target_frame)

    def update_video_slider(self, current_frame, total_frames):
        if total_frames > 0:
            val = int((current_frame / total_frames) * 1000)
            self.video_slider.blockSignals(True)
            self.video_slider.setValue(val)
            self.video_slider.blockSignals(False)

    def change_video_speed(self, val):
        if hasattr(self, 'video_thread'): self.video_thread.set_speed(val)

    def apply_video_filter(self, filter_name):
        if hasattr(self, 'video_thread'): self.video_thread.current_filter = filter_name

    def start_camera(self):
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
            self.camera_timer.start(30)

    def stop_camera(self):
        if self.camera_timer.isActive(): self.camera_timer.stop()
        if self.cap and self.cap.isOpened(): self.cap.release()
        self.cam_display_label.setText("تم إيقاف الكاميرا")

    def set_camera_mode(self, mode):
        self.camera_mode = mode

    def update_camera_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # تطبيق الفلاتر الخمسة المتقدمة للوجه والكاميرا
                frame = self.face_detector.detect_and_process(frame, self.camera_mode)
                self.current_cam_frame = frame.copy()
                self.display_cv_image(self.cam_display_label, frame)

    def capture_photo(self):
        if hasattr(self, 'current_cam_frame') and self.current_cam_frame is not None:
            path, _ = QFileDialog.getSaveFileName(self, "حفظ الصورة الملتقطة", "", "PNG (*.png);;JPEG (*.jpg)")
            if path:
                cv2.imwrite(path, self.current_cam_frame)
                QMessageBox.information(self, "نجاح", "تم التقاط الصورة وحفظها بنجاح!")

    def display_cv_image(self, label, cv_img):
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        
        target_size = label.size()
        if target_size.width() > 10 and target_size.height() > 10:
            label.setPixmap(pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def apply_stylesheet(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #0F172A; }
            #Sidebar { background-color: #1E293B; border-right: 1px solid #334155; }
            #Sidebar QPushButton {
                background-color: transparent;
                color: #94A3B8;
                border: none;
                text-align: right;
                padding-right: 20px;
                font-size: 14px;
                font-family: 'Segoe UI', 'Cairo', sans-serif;
            }
            #Sidebar QPushButton:hover {
                background-color: #334155;
                color: #FFFFFF;
            }
            QPushButton {
                background-color: #0284C7;
                color: white;
                border-radius: 6px;
                padding: 6px 12px;
                font-size: 13px;
                font-family: 'Segoe UI', 'Cairo', sans-serif;
            }
            QPushButton:hover { background-color: #0369A1; }
            QGroupBox {
                color: #38BDF8;
                font-weight: bold;
                border: 1px solid #334155;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
                font-family: 'Segoe UI', 'Cairo', sans-serif;
            }
        """)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PixelStudioApp()
    window.show()
    sys.exit(app.exec_())