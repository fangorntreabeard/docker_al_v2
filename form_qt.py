from PyQt6 import QtWidgets, QtCore
import sys
import json
from scripts.detection.train import train_api
from scripts.detection.eval import eval
from scripts.other.list_to_cocofile import write_json
import datetime

class MyWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.setWindowTitle('Активное обучение')
        # window.resize(600, 300)
        self.vbox = QtWidgets.QVBoxLayout()

        self.line_add = QtWidgets.QSpinBox()
        self.line_add.setMaximum(9999)
        self.line_add.setValue(500)
        self.lineEdit_a = QtWidgets.QLineEdit()
        self.lineEdit_a.setText('ds')

        self.lineEdit1 = QtWidgets.QLineEdit()
        self.lineEdit1.setText('##/coco/train2017')
        self.lineEdit2 = QtWidgets.QLineEdit()
        self.lineEdit2.setText('##/coco/labelstrain')
        self.lineEdit3 = QtWidgets.QLineEdit()
        self.lineEdit3.setText('##/coco/val2017')
        self.lineEdit4 = QtWidgets.QLineEdit()
        self.lineEdit4.setText('##/coco/labelsval')
        self.lineEdit5 = QtWidgets.QLineEdit()
        self.lineEdit5.setText('##/coco/boxes/')
        self.lineEdit6 = QtWidgets.QLineEdit()
        self.lineEdit6.setText('##/coco/classification/')
        self.form = QtWidgets.QFormLayout()

        self.form.addRow('Количество изображений:', self.line_add)
        self.form.addRow('Путь (##):', self.lineEdit_a)
        self.form.addRow('<b>Настройки для обучения:</b>', QtWidgets.QLabel())
        self.form.addRow('Путь до изображений тренировки:', self.lineEdit1)
        self.form.addRow('Путь до меток тренировки:', self.lineEdit2)
        self.form.addRow('<b>Настройки для валидации:</b>', QtWidgets.QLabel())
        self.form.addRow('Путь до изображений валидации:', self.lineEdit3)
        self.form.addRow('Путь до меток валидации:', self.lineEdit4)
        self.form.addRow('<b>Технические настройки:</b>', QtWidgets.QLabel())
        self.form.addRow('Путь для боксов:', self.lineEdit5)
        self.form.addRow('Путь для классификатора:', self.lineEdit6)

        self.button1 = QtWidgets.QPushButton('Запустить')
        self.button1.clicked.connect(self.click_start)

        # /home/neptun/PycharmProjects/datasets/:/ds
        self.help_label1 = QtWidgets.QLabel()
        self.help_label1.setText('1. Пути должны начитаться со слова <i style="color:#40E0D0";>ds</i>, '
                                 'при этом на хосте папка с датасетами располагается в '
                                 '/home/neptun/PycharmProjects/datasets.<br>'
                                 'Например, путь на хосте /home/neptun/PycharmProjects/datasets/coco/labelstrain, '
                                 'нужно записать как /ds/coco/labelstrain')

        self.help_label2 = QtWidgets.QLabel()
        self.help_label2.setText('2. В папке для классификатора необходимо сделать такую иерархию: <br>'
                                 'classification / train / be /; '
                                 'classification / train / notbe /; '
                                 'classification / val / be /; '
                                 'classification / val / notbe /; ')

        self.out_line = QtWidgets.QTextEdit('Изображения для разметки')
        self.button_save = QtWidgets.QPushButton('Save')
        self.button_save.clicked.connect(self.click_save)
        self.h2 = QtWidgets.QHBoxLayout()
        self.h2.addWidget(self.out_line)
        self.h2.addWidget(self.button_save, alignment=QtCore.Qt.AlignmentFlag.AlignTop)


        self.work_line = QtWidgets.QLabel('...')

        self.vbox.addLayout(self.form)
        self.vbox.addWidget(self.help_label1)
        self.vbox.addWidget(self.help_label2)
        self.vbox.addLayout(self.h2)
        self.vbox.addWidget(self.work_line)

        self.vbox.addWidget(self.button1)

        self.setLayout(self.vbox)

    def click_start(self):
        t_start = datetime.datetime.now()
        tot = self.lineEdit_a.text()

        path_to_labels = self.lineEdit2.text().replace('##', tot)
        path_to_img = self.lineEdit1.text().replace('##', tot)
        path_to_labels_val = self.lineEdit4.text().replace('##', tot)
        path_to_img_val = self.lineEdit3.text().replace('##', tot)
        path_to_boxes = self.lineEdit5.text().replace('##', tot)
        path_to_classes = self.lineEdit6.text().replace('##', tot)
        add = int(self.line_add.text())
        device_rest = 'gpu'

        self.work_line.setText('Расчет модели детектора')

        out = eval(path_to_labels, path_to_img, path_to_labels_val, path_to_img_val, device_rest)
        f, model = out['mAP(0.5:0.95)'], out['model']

        self.work_line.setText('mAP0 = {}'.format(f))

        step = train_api(path_to_img, path_to_labels, path_to_img_val, path_to_labels_val,
                         path_to_boxes, path_to_classes, add, device_rest, model)

        self.out_line.setText(json.dumps(step['data']))
        print('DONE {}'.format(datetime.datetime.now() - t_start))

    def click_save(self):
        list_files = json.loads(self.out_line.toPlainText())
        tot = self.lineEdit_a.text()

        m = 'default'
        s, ok = QtWidgets.QInputDialog.getText(self, 'Сохранение разметки', "Префикс", text='pref')
        if ok:
            m = s
        full_train_file = QtWidgets.QFileDialog.getOpenFileName(parent=self,
                                                                caption='Сохранение разметки',
                                                                filter='All (*);;JSON (*.json)',
                                                                initialFilter='JSON (*.json)')
        full_train_json = full_train_file[0]
        write_json(list_files, m, self.lineEdit2.text().replace('##', tot), full_train_json)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec())
