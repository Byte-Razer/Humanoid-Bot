try:
    from PyQt5 import QtCore, QtGui, QtWidgets
except ImportError:
    from PySide6 import QtCore, QtGui, QtWidgets
import sys, os, time, math, random, subprocess

# ----------------------------- Worker for background tasks -----------------------------
class WorkerSignals(QtCore.QObject):
    progress = QtCore.pyqtSignal(int)
    finished = QtCore.pyqtSignal()
    message = QtCore.pyqtSignal(str)

class FakeLongTask(QtCore.QRunnable):
    def __init__(self, duration=5, message_prefix="Working"):
        super().__init__()
        self.duration = duration
        self.signals = WorkerSignals()
        self.message_prefix = message_prefix

    def run(self):
        steps = max(10, int(self.duration * 5))
        for i in range(steps + 1):
            pct = int(i * 100 / steps)
            self.signals.progress.emit(pct)
            if i % (max(1, steps // 5)) == 0:
                self.signals.message.emit(f"{self.message_prefix}... {pct}%")
            time.sleep(self.duration / steps)
        self.signals.message.emit(f"{self.message_prefix} complete.")
        self.signals.finished.emit()


# ----------------------------- Space Background -----------------------------
class SpaceBackground(QtWidgets.QWidget):
    def __init__(self, star_count=600, parent=None):
        super().__init__(parent)
        self.star_count = star_count
        self.stars = []
        for _ in range(star_count):
            x = random.randint(0, 1920)
            y = random.randint(0, 1080)
            base_brightness = random.randint(180, 255)
            phase = random.random() * math.pi * 2
            self.stars.append([x, y, base_brightness, phase])
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(100)

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor(0, 0, 0))
        grad = QtGui.QLinearGradient(self.width() * 0.5, 0, self.width(), self.height())
        grad.setColorAt(0.0, QtGui.QColor(0, 0, 0, 0))
        grad.setColorAt(1.0, QtGui.QColor(100, 0, 160, 40))
        p.fillRect(self.rect(), grad)
        for x, y, base_brightness, phase in self.stars:
            brightness = base_brightness + 50 * math.sin(time.time() * 0.8 + phase)
            brightness = max(100, min(255, int(brightness)))
            p.setPen(QtGui.QColor(brightness, brightness, brightness))
            p.drawPoint(x % self.width(), y % self.height())


# ----------------------------- Glow Button -----------------------------
class GlowButton(QtWidgets.QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setFixedHeight(70)
        self.setMinimumWidth(300)
        self.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.setFocusPolicy(QtCore.Qt.NoFocus)

        self.shadow = QtWidgets.QGraphicsDropShadowEffect(self)
        self.shadow.setOffset(0, 0)
        self.shadow.setBlurRadius(0)
        self.shadow.setColor(QtGui.QColor(170, 0, 255, 160))
        self.setGraphicsEffect(self.shadow)

        self._scale = 1.0
        self.anim_group = QtCore.QParallelAnimationGroup(self)
        self.blur_anim = QtCore.QPropertyAnimation(self.shadow, b"blurRadius")
        self.scale_anim = QtCore.QPropertyAnimation(self, b"_scale_prop")

        for anim in (self.blur_anim, self.scale_anim):
            anim.setDuration(220)
            anim.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        self.anim_group.addAnimation(self.blur_anim)
        self.anim_group.addAnimation(self.scale_anim)

        self.setStyleSheet(self.base_stylesheet())

    def get_scale(self):
        return self._scale

    def set_scale(self, v):
        self._scale = v
        self.setStyleSheet(self.base_stylesheet(scale=v))

    _scale_prop = QtCore.pyqtProperty(float, fget=get_scale, fset=set_scale)

    def enterEvent(self, e):
        self.blur_anim.setStartValue(self.shadow.blurRadius())
        self.blur_anim.setEndValue(36)
        self.scale_anim.setStartValue(self._scale)
        self.scale_anim.setEndValue(1.03)
        self.anim_group.start()
        super().enterEvent(e)

    def leaveEvent(self, e):
        self.blur_anim.setStartValue(self.shadow.blurRadius())
        self.blur_anim.setEndValue(0)
        self.scale_anim.setStartValue(self._scale)
        self.scale_anim.setEndValue(1.0)
        self.anim_group.start()
        super().leaveEvent(e)

    def base_stylesheet(self, scale=1.0):
        # Futuristic deep blue → electric blue
        base1 = "rgba(10, 40, 90, 220)"     # deep blue, like original purple's dark base
        base2 = "rgba(25, 90, 160, 240)"    # lighter blue, like purple highlight

        hover1 = "rgba(20, 70, 140, 240)"   # hover brighter
        hover2 = "rgba(70, 150, 255, 255)"  # hover neon icy blue

        border_normal = "rgba(30,150,255,0.35)"
        border_hover  = "rgba(80,200,255,0.70)"

        text_color = "rgb(235,245,255)"

        return f"""
        QPushButton {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
            stop:0 {base1},
            stop:1 {base2});
        border: 2px solid {border_normal};
        border-radius: 18px;
        padding: 10px 18px;
        color: {text_color};
        font-size: 24px;
        font-weight: 600;
        letter-spacing: 0.6px;
        text-align: center;
        transform: scale({scale});
        }}
        QPushButton:hover {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
            stop:0 {hover1},
            stop:1 {hover2});
        border: 2px solid {border_hover};
        }}
        QPushButton:pressed {{
        transform: scale({max(0.98, scale - 0.02)});
        }}
        """
# ----------------------------- AURA Core -----------------------------
class AuraCore(QtWidgets.QLabel):
    def __init__(self):
        super().__init__()

        # Make widget larger so nothing clips
        self.setFixedSize(700, 700)

        # Make background fully transparent
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)

        self._color = QtGui.QColor(100,220,255)                                                                                      #100,220 255
        self._opacity = 180
        self._scale = 1.0
        self.phase = 0.0
        self.speed = 0.04

        # Saturn ring rotation
        self.saturn_angle = 0.0  

        self.anim_timer = QtCore.QTimer(self)
        self.anim_timer.timeout.connect(self.animate_pulse)
        self.anim_timer.start(30)

    def animate_pulse(self):
        # ORIGINAL orb animation
        self.phase += self.speed
        self._scale = 1.0 + 0.12 * math.sin(self.phase)
        self._opacity = 130 + 110 * (1 + math.sin(self.phase)) / 2

        # Rotate the white Saturn ring
        self.saturn_angle = (self.saturn_angle + 0.8) % 360

        self.update()

    def paintEvent(self, e):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)

        # Entire widget is now transparent
        p.fillRect(self.rect(), QtCore.Qt.transparent)

        # True orb size remains 320×320 centered inside 420×420
        orb_size = 520
        ox = (self.width() - orb_size) / 2
        oy = (self.height() - orb_size) / 2

        r = orb_size / 2
        cx = self.width() / 2
        cy = self.height() / 2

        # ---------------- ORIGINAL ORB ----------------
        p.save()
        p.translate(cx, cy)
        p.scale(self._scale, self._scale)
        p.translate(-cx, -cy)

        brightness_factor = 1.0 + ((self._scale - 1.0) * 2.5)
        brightness_factor = min(1.8, max(0.6, brightness_factor))

        glow_color = QtGui.QColor(
            min(int(self._color.red() * brightness_factor), 255),
            min(int(self._color.green() * brightness_factor), 255),
            min(int(self._color.blue() * brightness_factor), 255)
        )

        grad = QtGui.QRadialGradient(QtCore.QPointF(cx, cy), r)
        c = QtGui.QColor(glow_color)
        c.setAlpha(int(self._opacity))
        grad.setColorAt(0.0, c)
        grad.setColorAt(0.5, QtGui.QColor(c.red(), c.green(), c.blue(), 180))
        grad.setColorAt(0.8, QtGui.QColor(c.red(), c.green(), c.blue(), 80))
        grad.setColorAt(1.0, QtGui.QColor(0, 0, 0, 0))

        p.setBrush(grad)
        p.setPen(QtCore.Qt.NoPen)
        p.drawEllipse(QtCore.QRectF(ox, oy, orb_size, orb_size))
        p.restore()

        

    # ---------------- ORIGINAL pulse_react ----------------
    def pulse_react(self, color: QtGui.QColor):
        self._color = color
        anim = QtCore.QPropertyAnimation(self, b"geometry")
        anim.setDuration(600)
        start_rect = self.geometry()
        end_rect = QtCore.QRect(start_rect.x() - 25, start_rect.y() - 25,
                                start_rect.width() + 50, start_rect.height() + 50)
        anim.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        anim.setStartValue(start_rect)
        anim.setEndValue(end_rect)
        anim.finished.connect(lambda: self.reset_pulse(start_rect))
        anim.start(QtCore.QAbstractAnimation.DeleteWhenStopped)

    def reset_pulse(self, rect):
        anim = QtCore.QPropertyAnimation(self, b"geometry")
        anim.setDuration(800)
        anim.setEasingCurve(QtCore.QEasingCurve.InOutCubic)
        anim.setStartValue(self.geometry())
        anim.setEndValue(rect)
        anim.start(QtCore.QAbstractAnimation.DeleteWhenStopped)
# ----------------------------- Main Window -----------------------------
class AuraMain(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AURA — Interface")
        self.resize(1024, 600)

        self.background = SpaceBackground()
        self.setCentralWidget(self.background)

        self.overlay = QtWidgets.QWidget(self.background)
        self.overlay.setGeometry(self.background.rect())
        self.overlay.raise_()

        layout = QtWidgets.QVBoxLayout(self.overlay)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(20)
        layout.setAlignment(QtCore.Qt.AlignCenter)

        header = self.header_widget()
        layout.addWidget(header, alignment=QtCore.Qt.AlignHCenter)

        content = self.center_controls()
        layout.addWidget(content, alignment=QtCore.Qt.AlignHCenter)

        # Status label
        self.status_label = QtWidgets.QLabel("")
        self.status_label.setAlignment(QtCore.Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            font-size: 28px;
            font-weight: 600;
            color: rgba(150,200,255,0.8);
            letter-spacing: 2px;
            text-shadow: 0 0 25px rgba(100,160,255,0.6);
        """)
        layout.addWidget(self.status_label, alignment=QtCore.Qt.AlignHCenter)

        self.background.installEventFilter(self)
        self.threadpool = QtCore.QThreadPool()
        self.showMaximized()

    def eventFilter(self, s, e):
        if e.type() == QtCore.QEvent.Resize:
            self.overlay.setGeometry(self.background.rect())
        return super().eventFilter(s, e)

    # ----------------------------- HEADER (Title) -----------------------------
    def header_widget(self):
        w = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(w)

        # --- Load Centauri font ---
        from PyQt5.QtGui import QFontDatabase, QFont
        font_path = os.path.join(os.path.dirname(__file__), "Centauri", "Centauri.ttf")
        if not os.path.exists(font_path):
            font_path = None

        font_id = QFontDatabase.addApplicationFont(font_path)
        if font_id != -1:
            font_family = QFontDatabase.applicationFontFamilies(font_id)[0]
        else:
            font_family = "Segoe UI"  # fallback if font not found

        # --- AURA title label ---
        title = QtWidgets.QLabel("AURA")
        title.setFont(QFont(font_family, 110, QFont.Bold))
        title.setStyleSheet("""
            color: #1C6EDC;  /* bright but soft blue */
            letter-spacing: 10px;

            text-shadow:
                0 0 18px rgba(140, 200, 255, 0.8),   /* soft glow */
                0 0 42px rgba(100, 170, 255, 0.65),  /* faded shine */
                0 0 85px rgba(120, 190, 255, 0.50),  /* outer hazy glow */
                0 0 160px rgba(170, 220, 255, 0.35), /* hologram mist */
                0 0 240px rgba(200, 240, 255, 0.25); /* very soft fade */
        """)
        title.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)

        h.addWidget(title)
        h.addStretch()
        return w


    # ----------------------------- CENTER CONTROLS -----------------------------
    def center_controls(self):
        container = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(container)
        layout.setSpacing(300)
        layout.setContentsMargins(550,-250, 60, 40)

        self.aura_core = AuraCore()
        layout.addWidget(self.aura_core, alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)


        btn_col = QtWidgets.QVBoxLayout()
        btn_col.setSpacing(25)
        btn_col.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        self.btn_start = GlowButton("Start recognition")
        self.btn_train = GlowButton("Train data")
        self.btn_manage = GlowButton("Manage dataset")
        self.btn_listen = GlowButton("Listen")
        self.btn_exit = GlowButton("Exit")
        

        for b in [self.btn_start, self.btn_train, self.btn_manage, self.btn_listen]:
            btn_col.addWidget(b)

        self.btn_start.clicked.connect(self.start_recognition)
        self.btn_train.clicked.connect(self.train_data)
        self.btn_manage.clicked.connect(self.manage_dataset)
        self.btn_listen.clicked.connect(self.run_queries)
        self.btn_exit.clicked.connect(self.exit_app)


        # Add AURA core aligned to the left
        #layout.addWidget(self.aura_core, alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        # Add spacing between orb and buttons
        layout.addSpacing(300)  # adjust this gap visually 

        # Add the button column to the right
        layout.addLayout(btn_col, stretch=0)
        # --- Move Exit button fully to the bottom-right corner (anchorsed) ---
        self.btn_exit.setFixedSize(120, 45)
        self.btn_exit.setStyleSheet("""
        QPushButton {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 rgba(100,0,60,220), stop:1 rgba(200,0,100,255));
            border: 2px solid rgba(255,120,120,0.6);
            border-radius: 14px;
            color: rgb(255,230,230);
            font-size: 22px;
            font-weight: 6000;
        }
        QPushButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 rgba(150,0,80,240), stop:1 rgba(255,80,80,255));
        }
        """)

        # Create a floating container to hold the Exit button
        exit_container = QtWidgets.QWidget(self.overlay)
        exit_layout = QtWidgets.QHBoxLayout(exit_container)

        # ⬇️ Here's where you set the margins
        exit_layout.setContentsMargins(0, 0, 40, 40)  # right = 40px, bottom = 40px

        exit_layout.addStretch()
        exit_layout.addWidget(self.btn_exit, alignment=QtCore.Qt.AlignRight | QtCore.Qt.AlignBottom)
        exit_container.setLayout(exit_layout)

        # Anchor the container to always fill the parent (so it stays bottom-right automatically)
        #exit_container.setGeometry(0, 0, container.width(), container.height())
        # Fill overlay to anchor button properly
        exit_container.setGeometry(0, 0, self.overlay.width(), self.overlay.height())

        # ✅ Keep it on top of everything else (important!)
        exit_container.raise_()

        # Keep it updated when overlay resizes
        def overlay_resize(event):
            exit_container.setGeometry(0, 0, self.overlay.width(), self.overlay.height())

        self.overlay.resizeEvent = overlay_resize


        # Update geometry when window resizes
        def reposition_exit():
            exit_container.setGeometry(0, 0, container.width(), container.height())

        #container.resizeEvent = lambda event: reposition_exit()

        return container


    # ----------------------------- BUTTON ACTIONS -----------------------------
    def start_recognition(self):
        script_path = os.path.join(os.path.dirname(__file__), "recognise.py")
        if not os.path.exists(script_path):
            QtWidgets.QMessageBox.warning(self, "Error", "recognise.py not found in the app folder.")
            return
        try:
            # Visual feedback (blueish pulse)
            self.aura_core.pulse_react(QtGui.QColor(38, 103, 255))

            # Show "Recognizing..." in the status label
            self.status_label.setText("Recognizing...")

            # Run the recognition script
            subprocess.Popen([sys.executable, script_path])

            # Optionally clear the message after a few seconds
            QtCore.QTimer.singleShot(3000, lambda: self.status_label.clear())

        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", f"Failed to run recognise.py:\n{e}")
            print("Error while launching recognise.py:", e)


    def train_data(self):
        script_path = os.path.join(os.path.dirname(__file__), "train.py")
        if not os.path.exists(script_path):
            QtWidgets.QMessageBox.warning(self, "Error", "train.py not found in the app folder.")
            return

        # Ask for person name before training
        person_name, ok = QtWidgets.QInputDialog.getText(self, "Enter Name", "Enter the person's name for training:")
        if ok and person_name.strip():
            try:
                name = person_name.strip()

                # Pulse animation to show activity
                self.aura_core.pulse_react(QtGui.QColor(255, 220, 60))

                # Update GUI to show ongoing training
                self.status_label.setText(f"Registering {name}...")

                # Use QProcess instead of subprocess so we can detect completion
                self.training_process = QtCore.QProcess(self)
                self.training_process.finished.connect(lambda: self.training_done(name))

                # Start train.py
                self.training_process.start(sys.executable, [script_path, name])

            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Error", f"Failed to run train.py:\n{e}")

    def training_done(self, name):
        """Called when training process finishes."""
        self.aura_core.pulse_react(QtGui.QColor(80, 255, 120))
        self.status_label.setText(f"{name} Registration complete!")
        QtCore.QTimer.singleShot(3000, lambda: self.status_label.clear())
    def run_queries(self):
        """Launch queries.py when Listen is pressed."""
        script_path = os.path.join(os.path.dirname(__file__), "queries_api.py")

        if not os.path.exists(script_path):
            QtWidgets.QMessageBox.warning(self, "Error", f"queries.py not found at:\n{script_path}")
            return

        try:
            # visual feedback
            self.aura_core.pulse_react(QtGui.QColor(0, 255, 255))  # cyan pulse
            self.status_label.setText("Listening...")

            # run queries.py in a separate Python process
            subprocess.Popen([sys.executable, script_path])

            # clear status after a few seconds
            QtCore.QTimer.singleShot(3000, lambda: self.status_label.clear())

        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", f"Failed to run queries.py:\n{e}")
            print("Error while launching queries.py:", e)

    def manage_dataset(self):
        self.status_label.setText("Opening dataset folder...")
        self.aura_core.pulse_react(QtGui.QColor(180, 100, 255))

        data_path = os.path.join(os.path.dirname(__file__), "data")
        if not os.path.exists(data_path):
            os.makedirs(data_path, exist_ok=True)

        if sys.platform.startswith("win"):
            os.startfile(data_path)
        elif sys.platform.startswith("darwin"):
            subprocess.Popen(["open", data_path])
        else:
            subprocess.Popen(["xdg-open", data_path])

        QtCore.QTimer.singleShot(1500, lambda: self.status_label.clear())

    def exit_app(self):
        self.status_label.setText("Exiting...")
        self.aura_core.pulse_react(QtGui.QColor(255, 70, 70))
        reply = QtWidgets.QMessageBox.question(
            self, "Exit", "Exit AURA Interface?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        if reply == QtWidgets.QMessageBox.Yes:
            QtWidgets.QApplication.quit()
        else:
            self.status_label.clear()


# ----------------------------- Entrypoint -----------------------------
def main():
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    app = QtWidgets.QApplication(sys.argv)
    w = AuraMain()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":

    main()
