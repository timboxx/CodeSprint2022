from PyQt6 import uic
from PyQt6.QtWidgets import QApplication, QDialog, QMainWindow, QLabel, QVBoxLayout, QRadioButton, QWidget, QGraphicsView,QTextEdit, QLineEdit
from PyQt6.QtCore import pyqtSignal,pyqtSlot,Qt, QThread, QMetaObject, QCoreApplication
import PyQt6.QtGui as QtGui
from PyQt6.QtGui import QPixmap, QImage, QFont
import cv2
import numpy as np
import sys
import time
import os
from os import listdir
from gtts import gTTS
import os
import pyjokes
import logging
from PIL import Image, ImageDraw
from pprint import pprint
import boto3
from botocore.exceptions import ClientError
import io
from playsound import playsound
import asyncio
import shutil

from rekognition.rekognition_objects import  (
    RekognitionFace, RekognitionCelebrity, RekognitionLabel,
    RekognitionModerationLabel, RekognitionText, show_bounding_boxes, show_polygons)


from ui_UI_codesprint import Ui_MainWindow

defaultRegion = 'us-east-1'
defaultUrl = 'https://polly.us-east-1.amazonaws.com'

Form, Window = uic.loadUiType("UI_codesprint.ui")

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
    def run(self):
        # capture from web cam
        self.cap = cv2.VideoCapture(0)
        while True:
            ret, cv_img = self.cap.read()
            if ret:
                self.change_pixmap_signal.emit(cv_img)

class Window(QMainWindow,QWidget,Ui_MainWindow):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent=parent)

        self.menu_state = 0
        self.mute_state = 0
        self.mute_state = 1

        self.form = Form()
        self.form.setupUi(self)

        self.disply_width = 640
        self.display_height = 480

        n_channels = 4
        self.cv_img_buff = np.zeros(( self.disply_width , self.display_height, n_channels), dtype=np.uint8)
        self.speech_str = ""
        self.data_path = ""
        self.collection_path = ""
        self.names_list_path = ""
        self.Data_Destination_directory()
        self.image_path = self.data_path+"img_capture.png"

        # create the label that holds the image
        self.video_label = QLabel(self)
        self.video_label.setGeometry(0, 20, self.disply_width, self.display_height)
        # create a text label
        self.textLabel = QLabel('Webcam')
        # create the video capture thread
        try:
            self.thread = VideoThread()
            # connect its signal to the update_image slot
            self.thread.change_pixmap_signal.connect(self.update_video)

            # start the thread
            self.thread.start()
        except OSError as e:
            self.thread.join()

        self.image_result_label = QLabel(self)
        self.image_result_label.setGeometry(620, 520, 256, 192)
        # self.form.textEdit_2.setText("->Select Menu")

        QMetaObject.connectSlotsByName(self)

        self.form.radioButton.toggled.connect(lambda: self.select(self.form.radioButton))
        self.form.radioButton_2.toggled.connect(lambda: self.select(self.form.radioButton_2))
        self.form.radioButton_3.toggled.connect(lambda: self.select(self.form.radioButton_3))
        self.form.radioButton_4.toggled.connect(lambda: self.select(self.form.radioButton_4))
        self.form.radioButton_5.toggled.connect(lambda: self.select(self.form.radioButton_5))
        self.form.CaptureButton.clicked.connect(lambda: self.select(self.form.CaptureButton))
        self.form.checkBox_2.clicked.connect(lambda:self.mute_fn())
        self.form.LearnButton.clicked.connect(lambda:self.select(self.form.LearnButton))

        self.form.create_collection.clicked.connect(lambda: self.Create_collection())
        # self.form.

    def mute_fn(self):
        if self.form.checkBox_2.isChecked():
            self.mute_state = 1
        else:
            self.mute_state = 0

    def Data_Destination_directory(self):
        path = os.path.abspath(os.getcwd())
        new_dir = '/data/dst'
        new_path_image = path + new_dir
        self.data_path = new_path_image
        self.collection_path = new_path_image+ '/collection'
        check_exist = os.path.exists(new_path_image)

        if check_exist == True:
            pass
        else:
            try:
                os.makedirs(new_path_image)
            except OSError as error:
                print(error)

        check_exist = os.path.exists(self.collection_path)
        if check_exist == True:
            pass
        else:
            try:
                os.makedirs(self.collection_path)
            except OSError as error:
                print(error)
        # self.data_path_collection = new_path_image

    # Function State Handler
    def select(self,button):
        if button.isChecked():
            if button.text() == "FaceCognizer":
                self.menu_state = 1
                self.form.textEdit_2.setText("Have I seen you before.. ? ")
            if button.text() == "ItemAlyzer":
                self.menu_state = 2
                self.form.textEdit_2.setText("Finding Objects.. ")
            if button.text() == "FaceAlyzer":
                self.menu_state = 3
                self.form.textEdit_2.setText("Analysing Faces.. ")
            if button.text() == "RecoGnizer":
                self.menu_state = 4
                self.form.textEdit_2.setText("Reading Text.. ")
            if button.text() == "Auto":
                self.menu_state = 5
        if button.text() == "Learn":
            self.form.textEdit_2.setText("Learning..  "+str(self.form.lineEdit.text()))
        if button.text() == "Capture":
            asyncio.run(self.state_handler())

    async def state_handler(self):
        await self.Learn_name(self.form.lineEdit.text())
        await self.Capture_image()
        if self.menu_state == 0:
            self.form.textEdit_2.setText("->Select Menu")
        elif self.menu_state == 1:
            self.form.textEdit_2.setText("-> Capture Image")
            await self.FaceCognizer()
        elif self.menu_state == 2:
            self.form.textEdit_2.setText("-> Capture Image")
            await self.ItemAlyzer()
        elif self.menu_state == 3:
            self.form.textEdit_2.setText("-> Capture Image")
            await self.FaceAlyzer()
        elif self.menu_state == 4:
            self.form.textEdit_2.setText("-> Capture Image")
            await self.Recognise_text_func()
        elif self.menu_state == 5:
            await self.Auto_func()
        else:
            pass

    # Uses learnt faces to compare a captured image and identify individuals
    async def FaceCognizer(self):
        self.pyqt_Title("FaceCognizer")
        self.default_output_formatting()
        self.form.textEdit.append("Detecting...")
        await self.Capture_image()
        for images in os.listdir(self.collection_path):
            if (images.endswith(".jpg")):
                l_image_path = self.collection_path+'/'+ str(images)
                rek_image_compare = RekognitionImage.from_file(l_image_path,rekognition_client)
                rek_image_captured = RekognitionImage.from_file(self.data_path+str('/img_capture.jpg'), rekognition_client)
                matches, unmatches = rek_image_compare.compare_faces(rek_image_captured, 80)
                if len(matches) > 0:
                    recognized_person =images.rsplit(".", 1)[0]
                    self.form.textEdit.append('\t'+str(recognized_person))
                self.pyqt_show_boundary_box(rek_image_captured.image['Bytes'], [[match.bounding_box for match in matches]],['aqua'])

    # Reads text from a captured image
    async def Recognise_text_func(self):
        image_path = self.image_path
        self.pyqt_Title("RecoGnizer")
        self.default_output_formatting()
        text_image = RekognitionImage.from_file(image_path, rekognition_client)
        texts = text_image.detect_text()
        string_out_text = ""
        for text in texts:
            dict_out = text.to_dict()
            font = QFont()
            font.setPointSize(12)
            for key, value in dict_out.items():
                if key == 'text':
                    if type(value) == list:
                        value_joined = ",".join(value)
                        string_out_text = string_out_text + str(value_joined)
                        string_out_text = string_out_text + " "
                    elif type(value == str):
                        string_out_text = string_out_text + str(value)
                        string_out_text = string_out_text + " "
                    else:
                        pass

        self.form.textEdit.append(string_out_text)
        self.pyqt_show_polygons(text_image.image['Bytes'],[text.geometry['Polygon'] for text in texts])

        try:
            await self.speak(string_out_text,self.mute_state)
        except AssertionError:
            self.form.textEdit_2.setText('No text')
        finally:
            pass

    def Auto_func(self):
        print("g")


    async def FaceAlyzer(self):
        image_path = self.image_path
        try:
            self.pyqt_Title("FaceAlyzer")
            self.default_output_formatting()

            face_capture = RekognitionImage.from_file(
                image_path, rekognition_client)

            faces = face_capture.detect_faces()
            for face in faces[:3]:
                pprint(face.to_dict())

            for face in faces[:3]:
                dict_out = face.to_dict()
                # Font for Headers
                font = QFont()
                font.setPointSize(12)
                font.setBold(True)
                font.setUnderline(False)
                self.form.textEdit.setCurrentFont(font)

                self.form.textEdit.append("Attribute \t Value")
                font.setBold(False)
                self.form.textEdit.setCurrentFont(font)

                for key, value in dict_out.items():
                    if key == 'bounding_box':
                        pass
                    elif key == 'has':
                        value = "\n".join(value)
                        self.form.textEdit.append(value)

                    elif key == 'emotions':
                        value = ",".join(value)
                        self.form.textEdit.append(key + "\t" + value)
                    else:
                        self.form.textEdit.append(key + "\t" + str(value))
            # fix here
            self.pyqt_show_boundary_box(face_capture.image['Bytes'], [[face.bounding_box for face in faces]], 'red')

            image = cv2.imread(image_path)
            down_width = 256
            down_height = 192
            scale_down = (down_width,down_height)

            image = cv2.resize(image,scale_down, interpolation = cv2.INTER_LINEAR)

            resized_image_path = self.data_path + str("/img_capture_resized.jpg")
            cv2.imwrite(resized_image_path,image)
            pixmap = QPixmap(resized_image_path)
            self.image_result_label.setPixmap(pixmap)

        except (FileNotFoundError):
            self.form.textEdit_2.setText("->Capture Image")
        finally:
            pass

    # labels items on image buffer with 65% confidence
    async def ItemAlyzer(self):
        image_path = self.image_path
        try:
            self.pyqt_Title("ItemAlyzer")
            self.default_output_formatting()
            items_image = RekognitionImage.from_file(
                image_path, rekognition_client)
            print(f"Detecting labels in {items_image.image_name}...")
            labels = items_image.detect_labels(100)
            string_out_text = ""
            count = 1
            for label in labels:

                dict_out = label.to_dict()
                font = QFont()
                font.setPointSize(12)

                for key, value in dict_out.items():
                    if key == 'confidence':
                        if value > 65:
                            pass
                        else:
                            continue
                    if key == 'name':
                        if type(key) == list:
                            value_joined = ",".join(value)
                            string_out_text = string_out_text +str(count)+ "."+str(value_joined)
                            string_out_text = string_out_text + "\n"
                            count= count+1
                        elif type(value == str):
                            string_out_text = string_out_text +str(count)+ "."+ str(value)
                            string_out_text = string_out_text + "\n"
                            count = count + 1
                        else:
                            pass

                    else:
                        pass

            self.form.textEdit.append(string_out_text)
            await self.speak(string_out_text,self.mute_state)
        except (FileNotFoundError):
            self.form.textEdit_2.setText("->Capture Image")
        except Exception as e:
            print(e)
        finally:
            pass

    # Captures image and stores in a global buffer
    async def Capture_image(self):
        self.form.textEdit_2.setText("Image captured")
        self.image_path = self.data_path+str("/img_capture.jpg")
        cv2.imwrite(self.image_path, self.cv_img_buff)

    def Create_collection(self):
        self.names_list_path = self.collection_path + str('learned_names.txt')
        try:
            if os.path.exists(self.collection_path):
                shutil.rmtree(self.collection_path+'/')
            else:
                pass
            if os.path.exists(self.names_list_path):
                self.form.textEdit_2.setText('No learned names')
            self.Data_Destination_directory()
        except Exception as e:
            print(e)
        # with open(self.names_list_path, 'w') as f:
        #     f.write('')

    # def Delete_collection(self):
    async def Learn_name(self,l_name):
        self.names_list_path = self.collection_path + str('/learned_names.txt')
        if l_name != "":
            image_path_name = self.collection_path+ str('/'+str(l_name)+'.jpg')
            try:
                with open(image_path_name, 'a') as f:
                    l_name = l_name + "\n"
                    f.write(l_name)
            except Exception as e:
                self.form.textEdit_2.setText('Change name input')
            finally:
                pass
            await self.Capture_image()
            cv2.imwrite(image_path_name, self.cv_img_buff)
            self.form.textEdit_2.setText('Learned ' + l_name)
        else:
            self.form.textEdit_2.setText('Input text to learn')


    @pyqtSlot(np.ndarray)
    def update_video(self, cv_img):
        """Updates the image_label with a new opencv image"""
        self.cv_img_buff = cv_img
        qt_img = self.convert_cv_qt(cv_img)
        self.video_label.setPixmap(qt_img)

    # converts image numpy array to pixmap for pyqt
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.AspectRatioMode.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def serialize_speech_dict_key(self,speech_str,key):
        self.speech_str = self.speech_str + key
        self.speech_str = (str(self.speech_str) + " ")

    def serialize_speech_dict_value(self,speech_str,value,colors):
        self.speech_str = self.speech_str + value
        self.speech_str = (str(self.speech_str) + " ")

    def pyqt_show_polygons_and_box(self, image_bytes, datas,colors):
        # needs fixing
        image = Image.open(io.BytesIO(image_bytes))
        draw = ImageDraw.Draw(image)

        # polygons = datas
        box_sets = datas

        for boxes, color in zip(box_sets, colors):
            for box in boxes:
                left = image.width * box['Left']
                top = image.height * box['Top']
                right = (image.width * box['Width']) + left
                bottom = (image.height * box['Height']) + top
                draw.rectangle([left, top, right, bottom], width=3)

        image.save(self.image_path)
        image = cv2.imread(self.image_path)
        down_width = 256
        down_height = 192
        scale_down = (down_width, down_height)

        image = cv2.resize(image, scale_down, interpolation=cv2.INTER_LINEAR)

        resized_image_path = self.data_path + str("/img_capture_resized.jpg")
        cv2.imwrite(resized_image_path, image)
        pixmap = QPixmap(resized_image_path)
        self.image_result_label.setPixmap(pixmap)

    # formats title
    def pyqt_Title(self,text):
        font = QFont()
        font.setPointSize(16)
        font.setUnderline(True)
        font.setBold(True)
        self.form.textEdit.setCurrentFont(font)
        self.form.textEdit.setText("\n"+str(text)+"\n")

    # formats default text
    def default_output_formatting(self):
        font = QFont()
        font.setPointSize(12)
        font.setUnderline(False)
        font.setBold(False)
        self.form.textEdit.setCurrentFont(font)

    # maps polygons on a captured image USES PIL
    def pyqt_show_polygons(self, image_bytes, polygons):
        image = Image.open(io.BytesIO(image_bytes))
        draw = ImageDraw.Draw(image)
        for polygon in polygons:
            draw.polygon([
                (image.width * point['X'], image.height * point['Y']) for point in polygon],
                outline='aqua')

        image.save(self.image_path)
        image = cv2.imread(self.image_path)
        down_width = 256
        down_height = 192
        scale_down = (down_width, down_height)

        image = cv2.resize(image, scale_down, interpolation=cv2.INTER_LINEAR)

        resized_image_path = self.data_path + str("/img_capture_resized.jpg")
        cv2.imwrite(resized_image_path, image)
        pixmap = QPixmap(resized_image_path)
        self.image_result_label.setPixmap(pixmap)

    # maps rectangle on a captured image USES PIL
    def pyqt_show_boundary_box(self, image_bytes, box_sets,colors):
        image = Image.open(io.BytesIO(image_bytes))
        draw = ImageDraw.Draw(image)
        for boxes, color in zip(box_sets, colors):
            for box in boxes:
                left = image.width * box['Left']
                top = image.height * box['Top']
                right = (image.width * box['Width']) + left
                bottom = (image.height * box['Height']) + top
                draw.rectangle([left, top, right, bottom], width=3)

        image.save(self.image_path)
        image = cv2.imread(self.image_path)
        down_width = 256
        down_height = 192
        scale_down = (down_width, down_height)

        image = cv2.resize(image, scale_down, interpolation=cv2.INTER_LINEAR)

        resized_image_path = self.data_path + str("/img_capture_resized.jpg")
        cv2.imwrite(resized_image_path, image)
        pixmap = QPixmap(resized_image_path)
        self.image_result_label.setPixmap(pixmap)

    # parses text for speech engine
    async def speak(self,text,mute_state):
        if (mute_state == 1):
            pass
        else:
            audio = gTTS(text=text, lang="en", slow=False)
            audio.save("output.mp3")
            playsound("output.mp3",block=False)


logger = logging.getLogger(__name__)


# snippet-end:[python.example_code.rekognition.image_detection_imports]


# snippet-start:[python.example_code.rekognition.RekognitionImage]
class RekognitionImage:
    """
    Encapsulates an Amazon Rekognition image. This class is a thin wrapper
    around parts of the Boto3 Amazon Rekognition API.
    """

    def __init__(self, image, image_name, rekognition_client):
        """
        Initializes the image object.

        :param image: Data that defines the image, either the image bytes or
                      an Amazon S3 bucket and object key.
        :param image_name: The name of the image.
        :param rekognition_client: A Boto3 Rekognition client.
        """
        self.image = image
        self.image_name = image_name
        self.rekognition_client = rekognition_client

    # snippet-end:[python.example_code.rekognition.RekognitionImage]

    # snippet-start:[python.example_code.rekognition.RekognitionImage.from_file]
    @classmethod
    def from_file(cls, image_file_name, rekognition_client, image_name=None):
        """
        Creates a RekognitionImage object from a local file.

        :param image_file_name: The file name of the image. The file is opened and its
                                bytes are read.
        :param rekognition_client: A Boto3 Rekognition client.
        :param image_name: The name of the image. If this is not specified, the
                           file name is used as the image name.
        :return: The RekognitionImage object, initialized with image bytes from the
                 file.
        """
        with open(image_file_name, 'rb') as img_file:
            image = {'Bytes': img_file.read()}
        name = image_file_name if image_name is None else image_name
        return cls(image, name, rekognition_client)

    # snippet-end:[python.example_code.rekognition.RekognitionImage.from_file]

    # snippet-start:[python.example_code.rekognition.RekognitionImage.from_bucket]
    @classmethod
    def from_bucket(cls, s3_object, rekognition_client):
        """
        Creates a RekognitionImage object from an Amazon S3 object.

        :param s3_object: An Amazon S3 object that identifies the image. The image
                          is not retrieved until needed for a later call.
        :param rekognition_client: A Boto3 Rekognition client.
        :return: The RekognitionImage object, initialized with Amazon S3 object data.
        """
        image = {'S3Object': {'Bucket': s3_object.bucket_name, 'Name': s3_object.key}}
        return cls(image, s3_object.key, rekognition_client)

    # snippet-end:[python.example_code.rekognition.RekognitionImage.from_bucket]

    # snippet-start:[python.example_code.rekognition.DetectFaces]
    def detect_faces(self):
        """
        Detects faces in the image.

        :return: The list of faces found in the image.
        """
        try:
            response = self.rekognition_client.detect_faces(
                Image=self.image, Attributes=['ALL'])
            faces = [RekognitionFace(face) for face in response['FaceDetails']]
            logger.info("Detected %s faces.", len(faces))
        except ClientError:
            logger.exception("Couldn't detect faces in %s.", self.image_name)
            raise
        else:
            return faces

    # snippet-end:[python.example_code.rekognition.DetectFaces]

    # snippet-start:[python.example_code.rekognition.CompareFaces]
    def compare_faces(self, target_image, similarity):
        """
        Compares faces in the image with the largest face in the target image.

        :param target_image: The target image to compare against.
        :param similarity: Faces in the image must have a similarity value greater
                           than this value to be included in the results.
        :return: A tuple. The first element is the list of faces that match the
                 reference image. The second element is the list of faces that have
                 a similarity value below the specified threshold.
        """
        try:
            response = self.rekognition_client.compare_faces(
                SourceImage=self.image,
                TargetImage=target_image.image,
                SimilarityThreshold=similarity)
            matches = [RekognitionFace(match['Face']) for match
                       in response['FaceMatches']]
            unmatches = [RekognitionFace(face) for face in response['UnmatchedFaces']]
        except ClientError:
            logger.exception(
                "Couldn't match faces from %s to %s.", self.image_name,
                target_image.image_name)
            raise
        else:
            return matches, unmatches

    # snippet-end:[python.example_code.rekognition.CompareFaces]

    # snippet-start:[python.example_code.rekognition.DetectLabels]
    def detect_labels(self, max_labels):
        """
        Detects labels in the image. Labels are objects and people.

        :param max_labels: The maximum number of labels to return.
        :return: The list of labels detected in the image.
        """
        try:
            response = self.rekognition_client.detect_labels(
                Image=self.image, MaxLabels=max_labels)
            labels = [RekognitionLabel(label) for label in response['Labels']]
            # logger.info("Found %s labels in %s.", len(labels), self.image_name)
        except ClientError:
            logger.info("Couldn't detect labels in %s.", self.image_name)
            raise
        else:
            return labels

    # snippet-end:[python.example_code.rekognition.DetectLabels]

    # snippet-start:[python.example_code.rekognition.DetectModerationLabels]
    def detect_moderation_labels(self):
        """
        Detects moderation labels in the image. Moderation labels identify content
        that may be inappropriate for some audiences.

        :return: The list of moderation labels found in the image.
        """
        try:
            response = self.rekognition_client.detect_moderation_labels(
                Image=self.image)
            labels = [RekognitionModerationLabel(label)
                      for label in response['ModerationLabels']]
            logger.info(
                "Found %s moderation labels in %s.", len(labels), self.image_name)
        except ClientError:
            logger.exception(
                "Couldn't detect moderation labels in %s.", self.image_name)
            raise
        else:
            return labels

    # snippet-end:[python.example_code.rekognition.DetectModerationLabels]

    # snippet-start:[python.example_code.rekognition.DetectText]
    def detect_text(self):
        """
        Detects text in the image.

        :return The list of text elements found in the image.
        """
        try:
            response = self.rekognition_client.detect_text(Image=self.image)
            texts = [RekognitionText(text) for text in response['TextDetections']]
            logger.info("Found %s texts in %s.", len(texts), self.image_name)
        except ClientError:
            logger.exception("Couldn't detect text in %s.", self.image_name)
            raise
        else:
            return texts

    # snippet-end:[python.example_code.rekognition.DetectText]

    # snippet-start:[python.example_code.rekognition.RecognizeCelebrities]
    def recognize_celebrities(self):
        """
        Detects celebrities in the image.

        :return: A tuple. The first element is the list of celebrities found in
                 the image. The second element is the list of faces that were
                 detected but did not match any known celebrities.
        """
        try:
            response = self.rekognition_client.recognize_celebrities(
                Image=self.image)
            celebrities = [RekognitionCelebrity(celeb)
                           for celeb in response['CelebrityFaces']]
            other_faces = [RekognitionFace(face)
                           for face in response['UnrecognizedFaces']]
            logger.info(
                "Found %s celebrities and %s other faces in %s.", len(celebrities),
                len(other_faces), self.image_name)
        except ClientError:
            logger.exception("Couldn't detect celebrities in %s.", self.image_name)
            raise
        else:
            return celebrities, other_faces


if __name__ == '__main__':



    try:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        rekognition_client = boto3.client('rekognition')
        app = QApplication(sys.argv)
        window = Window()
        window.show()
        sys.exit(app.exec())
    except (KeyboardInterrupt,SystemExit):
        print("Exit Handled")
        cv2.destroyAllWindows()

