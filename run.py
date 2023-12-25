# -*-coding = utf-8 -*-
# @Time : 2023/12/4 16:22
# @Author : 万锦
# @File : run.py.py
# @Softwore : PyCharm

from PyQt5.QtGui import *
from qfluentwidgets import MessageBox
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import numpy as np
import sys
import os
import time
import pandas as pd
import math
from matplotlib import pyplot as plt
from ui.solar_energy import Ui_MainWindow

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


#地表净辐射计算
def rn(rns,rnl):
    rn = rns - rnl
    return rn

#温度转换
def tk(t):
    tk = t + 273.16
    return tk

#出射净长波辐射
def rnl(c,tmax,tmin,ea,rs,rs0):
    rnl = c*((pow(tmax,4)-pow(tmin,4))/4)*(0.34 - 0.14*pow(ea,0.5))*(1.35*rs/rs0 - 0.35)
    return rnl

#入射净短波辐射计算
def rns(rs):
    rns = 0.77*rs
    return rns

#太阳短波辐射
def rs(ssd,n,ra):
    rs = (0.25 + 0.5*ssd/n)*ra
    return rs

#晴空辐射
def rs0(z,ra):
    rs0 = (0.75 + 2*pow(10,-5)*z)*ra
    return rs0

#大气层顶太阳辐射
#sun_angle太阳时角
def ra(weidu,j):
    weidu = weidu*(math.pi/180)
    #计算太阳磁偏角
    sun_cipianjiao = 0.409*math.sin(2*math.pi*j/365 - 1.39)
    #计算太阳时角
    sun_angle = math.acos(-math.tan(weidu)*math.tan(sun_cipianjiao))
    dr = 1+0.033*math.cos(2*math.pi*j/365)
    ra = 24*60/math.pi*0.0820*dr*(sun_angle*math.sin(weidu)*math.sin(sun_cipianjiao) + math.cos(weidu)*math.cos(sun_cipianjiao)*math.sin(sun_angle))
    return ra

#实际水汽压
def ea(rh,ta,tmax,tmin):
    tmax = tk(tmax)
    tmin = tk(tmin)
    e0 = 0.6108 * math.exp(17.27*ta/(ta+237.3))
    ea = rh/100*((e0*tmax+e0*tmin)/2)
    return ea
def N(weidu,j):
    weidu = weidu * (math.pi / 180)
    sun_cipianjiao = 0.409 * math.sin(2 * math.pi * j / 365 - 1.39)
    sun_angle = math.acos(-math.tan(weidu) * math.tan(sun_cipianjiao))
    n = 24 / math.pi * sun_angle

    return n


class Form_waterinf(QMainWindow,Ui_MainWindow):
    def __init__(self):
        super(Form_waterinf, self).__init__()
        self.setupUi(self)
        self.setObjectName("solar_energy")
        self.handlebutton()
        self.ininitialize()
        self.year_runoff = []
        self.time_series = []

    def ininitialize(self):
        self.PushButton_2.setEnabled(False)
        self.PushButton_3.setEnabled(False)
        self.PushButton_4.setEnabled(False)
        # self.PushButton_5.setEnabled(False)

    def handlebutton(self):
        self.PushButton.clicked.connect(self.add_data)
        self.PushButton.clicked.connect(self.deal_button)
        self.PushButton_7.clicked.connect(self.add_data)
        self.PushButton_8.clicked.connect(self.add_data)
        self.PushButton_9.clicked.connect(self.add_data)
        self.PushButton_10.clicked.connect(self.add_data)
        self.PushButton_11.clicked.connect(self.add_data)
        self.PushButton_12.clicked.connect(self.add_data)
        # self.PushButton_12.clicked.connect(self.test_def)
        # self.PushButton_2.clicked.connect(self.mk_test_mutation)
        # self.PushButton_3.clicked.connect(self.pettitt_test)
        # self.PushButton_4.clicked.connect(self.agglomerative)
        # self.PushButton_5.clicked.connect(self.contive_analysis)
        self.PushButton_6.clicked.connect(self.call_author)

    #解除按钮限制
    def deal_button(self):
        self.PushButton_2.setEnabled(True)
        self.PushButton_3.setEnabled(True)
        self.PushButton_4.setEnabled(True)
        # self.PushButton_5.setEnabled(True)

    # 测试函数
    def test_def(self):
        clicked_button_text = self.sender().text()
        print(f'Button clicked: {clicked_button_text}')
    #导入数据
    def add_data(self):
        fname, _ = QFileDialog.getOpenFileName(self, "打开文件", '.', '数据文件(*.txt)')
        clicked_button_text = self.sender().text()
        if fname:
            df = pd.read_csv(fname, index_col=0)
            if clicked_button_text == '载入高程':
                self.df_height = df
            if clicked_button_text == '载入平均气温':
                self.df_tmean = df
            if clicked_button_text == '载入最高气温':
                self.df_tmax = df
            if clicked_button_text == '载入最低气温':
                self.df_tmin = df
            if clicked_button_text == '载入纬度':
                self.df_weidu = df
            if clicked_button_text == '载入日照时数':
                self.df_sun = df
            if clicked_button_text == '载入湿度':
                self.df_rhu = df
            data = df.values
            # 表格加载数据
            # 设置行列，设置表头
            tmp = df.columns
            self.time_series = df.index.values
            tmp2 = [str(_) for _ in df.index.tolist()]
            self.TableWidget.setRowCount(len(data))
            self.TableWidget.setColumnCount(len(data[0]))
            self.TableWidget.setHorizontalHeaderLabels(tmp)
            self.TableWidget.setVerticalHeaderLabels(tmp2)
            # 表格加载内容
            for row, form in enumerate(data):
                for column, item in enumerate(form):
                    self.TableWidget.setItem(row, column, QTableWidgetItem(str(item)))


    #联系作者
    def call_author(self):
        title = '联系作者'
        content = """wanjinhhu@gmail.com"""
        # w = MessageDialog(title, content, self)   # Win10 style message box
        w = MessageBox(title, content, self)
        if w.exec():
           pass
        else:
            pass

def main():
    app = QApplication(sys.argv)
    mainwindow = Form_waterinf()
    mainwindow.setWindowTitle("太阳能资源评估系统V1.0")
    mainwindow.setWindowIcon(QIcon("./icons/三峡.ico"))
    mainwindow.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

