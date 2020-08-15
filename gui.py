# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 22:00:26 2020

@author: charleneboo
"""
filepath = "C://Users//65972//Desktop//term 8 studies//AI//ai proj//flatland//examples"
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel
from PyQt5.QtCore import QProcess
app = QApplication([])
window = QWidget()
layout = QVBoxLayout()

yearly_income = QLabel()
yearly_income.setText('Yearly Income: $0.00')
layout.addWidget(yearly_income)

tax_rate = QLabel()
tax_rate.setText('Highest Marginal Tax Rate: 0%')
layout.addWidget(tax_rate)

confirm_button = QPushButton('Confirm')
def confirm_event(): 
    QProcess.startDetached(filepath)
button.clicked.connect(confirm_event)layout.addWidget(confirm_button)
layout.addWidget(QPushButton('Bottom'))
window.setLayout(layout)
window.show()
# class Window(QMainWindow):

#     def __init__(self):
#         super(Window, self).__init__()
#         self.setGeometry(50, 50, 500, 300)
#         self.setWindowTitle("TEMP FILE")

#         self.home()

#     def home (self):
#         btn_run = QPushButton("Run", self)
#         btn_run.clicked.connect(self.execute)

#         self.show()

#     def execute(self):
#         subprocess.Popen('test1.py', shell=True)
#         subprocess.call(["python", "test1.py"])

# if not QtWidgets.QApplication.instance():
#     app = QtWidgets.QApplication(sys.argv)
# else:
#     app = QtWidgets.QApplication.instance()

# GUI = Window()
app.exec_()