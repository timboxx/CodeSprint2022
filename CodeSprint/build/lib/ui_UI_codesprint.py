# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'UI_codesprintpuOArB.ui'
##
## Created by: Qt User Interface Compiler version 5.15.5
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *  # type: ignore
from PySide2.QtGui import *  # type: ignore
from PySide2.QtWidgets import *  # type: ignore


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1307, 767)
        self.actionItemAlyzer = QAction(MainWindow)
        self.actionItemAlyzer.setObjectName(u"actionItemAlyzer")
        self.actionItemAlyzer.setCheckable(True)
        self.actionFaceCognizer = QAction(MainWindow)
        self.actionFaceCognizer.setObjectName(u"actionFaceCognizer")
        self.actionFaceCognizer.setCheckable(True)
        self.actionFaceAlyzer = QAction(MainWindow)
        self.actionFaceAlyzer.setObjectName(u"actionFaceAlyzer")
        self.actionFaceAlyzer.setCheckable(True)
        self.actionRecognise_Text = QAction(MainWindow)
        self.actionRecognise_Text.setObjectName(u"actionRecognise_Text")
        self.actionRecognise_Text.setCheckable(True)
        self.actionAuto_button = QAction(MainWindow)
        self.actionAuto_button.setObjectName(u"actionAuto_button")
        self.actionAuto_button.setCheckable(True)
        self.actionCapture_button = QAction(MainWindow)
        self.actionCapture_button.setObjectName(u"actionCapture_button")
        self.actionCapture_button.setCheckable(True)
        self.actionMute_toggle = QAction(MainWindow)
        self.actionMute_toggle.setObjectName(u"actionMute_toggle")
        self.actionMute_toggle.setCheckable(True)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.frame = QFrame(self.centralwidget)
        self.frame.setObjectName(u"frame")
        self.frame.setGeometry(QRect(0, 0, 881, 751))
        self.frame.setFrameShape(QFrame.StyledPanel)
        self.frame.setFrameShadow(QFrame.Raised)
        self.label = QLabel(self.frame)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(20, 520, 131, 18))
        font = QFont()
        font.setPointSize(16)
        self.label.setFont(font)
        self.lineEdit = QLineEdit(self.frame)
        self.lineEdit.setObjectName(u"lineEdit")
        self.lineEdit.setGeometry(QRect(20, 540, 141, 32))
        self.lineEdit.setAutoFillBackground(False)
        self.lineEdit.setStyleSheet(u"background-color: rgb(170, 255, 255);\n"
"color: rgb(0, 0, 0);")
        self.LearnButton = QPushButton(self.frame)
        self.LearnButton.setObjectName(u"LearnButton")
        self.LearnButton.setGeometry(QRect(170, 540, 84, 34))
        self.create_collection = QPushButton(self.frame)
        self.create_collection.setObjectName(u"create_collection")
        self.create_collection.setGeometry(QRect(10, 600, 111, 34))
        self.create_collection.setMouseTracking(True)
        self.create_collection.setStyleSheet(u"background-color: rgb(0, 0, 127);")
        self.delete_collection = QPushButton(self.frame)
        self.delete_collection.setObjectName(u"delete_collection")
        self.delete_collection.setGeometry(QRect(130, 600, 111, 34))
        self.delete_collection.setMouseTracking(True)
        self.delete_collection.setStyleSheet(u"background-color: rgb(170, 0, 0);")
        self.CaptureButton = QPushButton(self.frame)
        self.CaptureButton.setObjectName(u"CaptureButton")
        self.CaptureButton.setGeometry(QRect(400, 530, 84, 34))
        self.CaptureButton.setMouseTracking(True)
        self.CaptureButton.setStyleSheet(u"background-color: rgb(0, 170, 0);")
        self.textEdit_2 = QTextEdit(self.frame)
        self.textEdit_2.setObjectName(u"textEdit_2")
        self.textEdit_2.setGeometry(QRect(20, 650, 181, 41))
        self.textEdit_2.setStyleSheet(u"color: rgb(0, 0, 0);\n"
"background-color: rgb(255, 255, 255);")
        self.textEdit_2.setReadOnly(True)
        self.checkBox_2 = QCheckBox(self.frame)
        self.checkBox_2.setObjectName(u"checkBox_2")
        self.checkBox_2.setGeometry(QRect(300, 520, 85, 22))
        self.checkBox_2.setChecked(True)
        self.radioButton = QRadioButton(self.centralwidget)
        self.radioButton.setObjectName(u"radioButton")
        self.radioButton.setGeometry(QRect(120, 490, 101, 22))
        self.radioButton_2 = QRadioButton(self.centralwidget)
        self.radioButton_2.setObjectName(u"radioButton_2")
        self.radioButton_2.setGeometry(QRect(10, 490, 101, 22))
        self.radioButton_3 = QRadioButton(self.centralwidget)
        self.radioButton_3.setObjectName(u"radioButton_3")
        self.radioButton_3.setGeometry(QRect(240, 490, 101, 22))
        self.radioButton_4 = QRadioButton(self.centralwidget)
        self.radioButton_4.setObjectName(u"radioButton_4")
        self.radioButton_4.setGeometry(QRect(350, 490, 121, 22))
        self.radioButton_5 = QRadioButton(self.centralwidget)
        self.radioButton_5.setObjectName(u"radioButton_5")
        self.radioButton_5.setGeometry(QRect(480, 490, 101, 22))
        self.textEdit = QTextEdit(self.centralwidget)
        self.textEdit.setObjectName(u"textEdit")
        self.textEdit.setGeometry(QRect(890, 110, 371, 411))
        self.textEdit.setStyleSheet(u"color: rgb(0, 0, 0);\n"
"background-color: rgb(255, 255, 255);")
        self.textEdit.setReadOnly(True)
        MainWindow.setCentralWidget(self.centralwidget)
        self.frame.raise_()
        self.radioButton.raise_()
        self.radioButton_5.raise_()
        self.radioButton_2.raise_()
        self.radioButton_4.raise_()
        self.radioButton_3.raise_()
        self.textEdit.raise_()
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1307, 30))
        self.menuCodeSprint_2022 = QMenu(self.menubar)
        self.menuCodeSprint_2022.setObjectName(u"menuCodeSprint_2022")
        MainWindow.setMenuBar(self.menubar)

        self.menubar.addAction(self.menuCodeSprint_2022.menuAction())

        self.retranslateUi(MainWindow)
        self.radioButton.pressed.connect(MainWindow.update)
        self.radioButton_2.pressed.connect(MainWindow.update)
        self.radioButton_4.pressed.connect(MainWindow.update)
        self.radioButton_3.pressed.connect(MainWindow.update)
        self.radioButton_5.pressed.connect(MainWindow.update)
        self.CaptureButton.clicked.connect(MainWindow.update)
        self.LearnButton.clicked.connect(MainWindow.update)
        self.radioButton.toggled.connect(self.frame.setEnabled)
        self.radioButton_2.toggled.connect(self.frame.setEnabled)
        self.radioButton_3.toggled.connect(self.frame.setEnabled)
        self.radioButton_4.toggled.connect(self.frame.setEnabled)
        self.checkBox_2.toggled.connect(self.checkBox_2.update)
        self.delete_collection.clicked.connect(self.delete_collection.update)
        self.LearnButton.clicked.connect(self.lineEdit.update)
        self.create_collection.clicked.connect(self.create_collection.update)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"RecogNice", None))
        self.actionItemAlyzer.setText(QCoreApplication.translate("MainWindow", u"ItemAlyzer", None))
        self.actionFaceCognizer.setText(QCoreApplication.translate("MainWindow", u"FaceCognizer", None))
        self.actionFaceAlyzer.setText(QCoreApplication.translate("MainWindow", u"FaceAlyzer", None))
        self.actionRecognise_Text.setText(QCoreApplication.translate("MainWindow", u"Recognise_Text", None))
        self.actionAuto_button.setText(QCoreApplication.translate("MainWindow", u"Auto_button", None))
        self.actionCapture_button.setText(QCoreApplication.translate("MainWindow", u"Capture_button", None))
        self.actionMute_toggle.setText(QCoreApplication.translate("MainWindow", u"Mute_toggle", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"Learn Face", None))
        self.lineEdit.setText("")
        self.lineEdit.setPlaceholderText(QCoreApplication.translate("MainWindow", u"John Doe", None))
        self.LearnButton.setText(QCoreApplication.translate("MainWindow", u"Learn", None))
        self.create_collection.setText(QCoreApplication.translate("MainWindow", u"Create Collection", None))
        self.delete_collection.setText(QCoreApplication.translate("MainWindow", u"Delete Collection", None))
        self.CaptureButton.setText(QCoreApplication.translate("MainWindow", u"Capture", None))
        self.checkBox_2.setText(QCoreApplication.translate("MainWindow", u"Mute", None))
        self.radioButton.setText(QCoreApplication.translate("MainWindow", u"FaceCognizer", None))
        self.radioButton_2.setText(QCoreApplication.translate("MainWindow", u"ItemAlyzer", None))
        self.radioButton_3.setText(QCoreApplication.translate("MainWindow", u"FaceAlyzer", None))
        self.radioButton_4.setText(QCoreApplication.translate("MainWindow", u"RecoGnizer", None))
        self.radioButton_5.setText(QCoreApplication.translate("MainWindow", u"Auto", None))
        self.menuCodeSprint_2022.setTitle(QCoreApplication.translate("MainWindow", u"CodeSprint 2022", None))
    # retranslateUi

