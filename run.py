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
from scipy.stats import linregress
import re
from ui.solar_energy import Ui_MainWindow

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 进行线性回归分析
def linear_regression(time_series, runoff_data):
    # time_series年份输入，runoff_data年均径流输入

    slope, intercept, r_value, p_value, std_err = linregress(time_series, runoff_data)
    return slope, intercept, r_value, p_value, std_err
#mk突变分析
def Kendall_change_point_detection(index,inputdata):
    inputdata = np.array(inputdata)
    n=inputdata.shape[0]
    Sk = [0]
    UFk = [0]
    s =  0
    Exp_value = [0]
    Var_value = [0]
    for i in range(1,n):
        for j in range(i):
            if inputdata[i] > inputdata[j]:
                s = s+1
            else:
                s = s+0
        Sk.append(s)
        Exp_value.append((i+1)*(i+2)/4 )                     # Sk[i]的均值
        Var_value.append((i+1)*i*(2*(i+1)+5)/72 )            # Sk[i]的方差
        UFk.append((Sk[i]-Exp_value[i])/np.sqrt(Var_value[i]))
    Sk2 = [0]
    UBk = [0]
    UBk2 = [0]
    # s归0
    s2 =  0
    Exp_value2 = [0]
    Var_value2 = [0]
    # 按时间序列逆转样本y
    inputdataT = list(reversed(inputdata))
    for i in range(1,n):
        for j in range(i):
            if inputdataT[i] > inputdataT[j]:
                s2 = s2+1
            else:
                s2 = s2+0
        Sk2.append(s2)
        Exp_value2.append((i+1)*(i+2)/4 )                     # Sk[i]的均值
        Var_value2.append((i+1)*i*(2*(i+1)+5)/72 )            # Sk[i]的方差
        UBk.append((Sk2[i]-Exp_value2[i])/np.sqrt(Var_value2[i]))
        UBk2.append(-UBk[i])
    UBkT = list(reversed(UBk2))
    diff = np.array(UFk) - np.array(UBkT)
    K    = list()
    # 找出交叉点
    for k in range(1,n):
        if diff[k-1]*diff[k]<0:
            K.append(k)

    return K,UFk,UBkT

#判断是否为日期格式，确定是否需要手动校正日期
def is_valid_date(input_str):
    pattern = r'^\d{4}-\d{2}-\d{2}$'  # 定义日期格式的正则表达式

    if re.match(pattern, input_str):
        return True
    else:
        return False

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
    rnl = c*((pow(tmax,4)-pow(tmin,4))/2)*(0.34 - 0.14*pow(ea,0.5))*(1.35*rs/rs0 - 0.35)
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
        self.inin_shuxing()
        self.handlebutton()
        self.ininitialize()
        self.year_runoff = []
        self.time_series = []

    def inin_shuxing(self):
        self.ZhDatePicker.setDate(QDate.currentDate())

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
        self.PushButton_13.clicked.connect(self.add_data)
        self.PushButton_15.clicked.connect(self.add_data)
        # self.PushButton_12.clicked.connect(self.test_def)
        # self.PushButton_2.clicked.connect(self.mk_test_mutation)
        # self.PushButton_3.clicked.connect(self.pettitt_test)
        # self.PushButton_4.clicked.connect(self.agglomerative)
        # self.PushButton_5.clicked.connect(self.contive_analysis)
        self.PushButton_6.clicked.connect(self.call_author)
        self.PushButton_2.clicked.connect(self.calculate_solar_energy)
        self.PushButton_3.clicked.connect(self.calculate_stability)
        self.PushButton_4.clicked.connect(self.trend_analysis)
        self.PushButton_5.clicked.connect(self.mutation_analysis)
        self.PushButton_16.clicked.connect(self.avg_trend_analysis)
        self.PushButton_17.clicked.connect(self.avg_mutation_analysis)



    #解除按钮限制
    def deal_button(self):
        self.PushButton_2.setEnabled(True)
        self.PushButton_3.setEnabled(True)
        self.PushButton_4.setEnabled(True)
        self.PushButton_5.setEnabled(True)

    # 测试函数
    def test_def(self):
        # clicked_button_text = self.sender().text()
        # print(f'Button clicked: {clicked_button_text}')
        self.statusBar().showMessage("test")

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
            if clicked_button_text == '载入J值':
                self.df_j = df
            if clicked_button_text == '载入年均辐射':
                self.df_avg = df

            self.SpinBox.setValue(int(len(df.iloc[:, 0]) / 365))
            if is_valid_date(str(df.index[0])):
                qdate = QDate.fromString(df.index[0], 'yyyy-MM-dd')
                self.ZhDatePicker.setDate(qdate)
            else:
                self.showDialog()

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

    #计算太阳辐射
    def calculate_solar_energy(self):

        self.statusBar().showMessage("正在计算太阳辐射，请勿操作！")
        row_lst = []
        col_lst = []
        for row in self.df_weidu.index.values:
            col_lst = []
            for col in self.df_weidu.columns.values:
                tmp_value = N(self.df_weidu.loc[row, col], self.df_j.loc[row, col])
                col_lst.append(tmp_value)
            row_lst.append(col_lst)
        self.df_n = pd.DataFrame(row_lst)
        self.df_n.index = self.df_rhu.index
        self.df_n.columns = self.df_rhu.columns

        # 温度单位转换
        df_tmax = self.df_tmax / 10
        df_tmin = self.df_tmin / 10
        df_tmean = self.df_tmean / 10

        # 计算相对湿度
        row_lst = []
        col_lst = []
        for row in self.df_rhu.index.values:
            col_lst = []
            for col in self.df_rhu.columns.values:
                tmp_value = ea(self.df_rhu.loc[row, col], self.df_tmean.loc[row, col],
                               self.df_tmax.loc[row, col], self.df_tmin.loc[row, col])
                col_lst.append(tmp_value)
            row_lst.append(col_lst)
        self.df_ea = pd.DataFrame(row_lst)
        self.df_ea.index = self.df_rhu.index
        self.df_ea.columns = self.df_rhu.columns

        # 计算大气层顶太阳辐射ra
        row_lst = []
        col_lst = []
        for row in self.df_weidu.index.values:
            col_lst = []
            for col in self.df_weidu.columns.values:
                tmp_value = ra(self.df_weidu.loc[row, col], self.df_j.loc[row, col])
                col_lst.append(tmp_value)
            row_lst.append(col_lst)
        self.df_ra = pd.DataFrame(row_lst)
        self.df_ra.index = self.df_rhu.index
        self.df_ra.columns = self.df_rhu.columns

        # 计算晴空辐射rs0
        row_lst = []
        col_lst = []
        for row in self.df_weidu.index.values:
            col_lst = []
            for col in self.df_weidu.columns.values:
                tmp_value = rs0(self.df_height.loc[row, col], self.df_ra.loc[row, col])
                col_lst.append(tmp_value)
            row_lst.append(col_lst)
        self.df_rs0 = pd.DataFrame(row_lst)
        self.df_rs0.index = self.df_rhu.index
        self.df_rs0.columns = self.df_rhu.columns

        # 太阳短波辐射rs
        row_lst = []
        col_lst = []
        for row in self.df_weidu.index.values:
            col_lst = []
            for col in self.df_weidu.columns.values:
                tmp_value = rs(self.df_sun.loc[row, col], self.df_n.loc[row, col], self.df_ra.loc[row, col])
                col_lst.append(tmp_value)
            row_lst.append(col_lst)
        self.df_rs = pd.DataFrame(row_lst)
        self.df_rs.index = self.df_rhu.index
        self.df_rs.columns = self.df_rhu.columns

        # 入射净短波辐射rns
        row_lst = []
        col_lst = []
        for row in self.df_weidu.index.values:
            col_lst = []
            for col in self.df_weidu.columns.values:
                tmp_value = rns(self.df_rs.loc[row, col])
                col_lst.append(tmp_value)
            row_lst.append(col_lst)
        self.df_rns = pd.DataFrame(row_lst)
        self.df_rns.index = self.df_rhu.index
        self.df_rns.columns = self.df_rhu.columns

        # 出射净长波辐射rnl
        row_lst = []
        col_lst = []
        # c 为Stefan-Boltzmann常数
        c = 4.903 * pow(10, -9)
        for row in self.df_weidu.index.values:
            col_lst = []
            for col in self.df_weidu.columns.values:
                tmp_value = rnl(c, self.df_tmax.loc[row, col], self.df_tmin.loc[row, col],
                                self.df_ea.loc[row, col], self.df_rs.loc[row, col],
                                self.df_rs0.loc[row, col])
                col_lst.append(tmp_value)
            row_lst.append(col_lst)
        self.df_rnl = pd.DataFrame(row_lst)
        self.df_rnl.index = self.df_rhu.index
        self.df_rnl.columns = self.df_rhu.columns

        # 地表净辐射rn
        self.df_rn = self.df_rns - self.df_rnl
        self.df_rn = self.df_rn.round(2)

        data = self.df_rn.values
        # 表格加载数据
        # 设置行列，设置表头
        tmp = self.df_rn.columns
        tmp2 = [str(_) for _ in self.df_rn.index.tolist()]
        self.TableWidget.setRowCount(len(data))
        self.TableWidget.setColumnCount(len(data[0]))
        self.TableWidget.setHorizontalHeaderLabels(tmp)
        self.TableWidget.setVerticalHeaderLabels(tmp2)
        # 表格加载内容
        for row, form in enumerate(data):
            for column, item in enumerate(form):
                self.TableWidget.setItem(row, column, QTableWidgetItem(str(item)))
        self.statusBar().showMessage(" ")

    #计算太阳能资源稳定度
    def calculate_stability(self):
        # date_str = self.ZhDatePicker.date().toString('yyyy-MM-dd')
        date_str = self.ZhDatePicker.getDate().toString('yyyy-MM-dd')
        # print(date_str)

        prod = int(self.SpinBox.text())*12

        empty = len(self.df_sun.iloc[:,0])

        # 计算每列的空值数量
        null_counts = self.df_sun.isnull().sum()

        # 获取空值大于 1000 的列名
        columns_to_drop = null_counts[null_counts > empty].index

        # 删除对应的列
        df_filtered = self.df_sun.drop(columns=columns_to_drop)
        # print(df_filtered)

        # 根据年份和月份分组，然后对每列应用 lambda 函数统计大于 60 的个数
        df_filtered.index = pd.to_datetime(df_filtered.index)
        result = df_filtered.groupby([df_filtered.index.year, df_filtered.index.month]).apply(lambda x: (x > 60).sum())
        result = pd.DataFrame(result.values)
        result.columns = df_filtered.columns
        result.index = pd.date_range(start=date_str, periods=prod, freq='M')

        # 按照年份分组，然后计算各列每年的最大值
        yearly_max_values = result.groupby(result.index.year).max()
        # 按照年份分组，然后计算各列每年的最小值
        yearly_min_values = result.groupby(result.index.year).min()

        self.yearly_mean_values = result.groupby(result.index.year).mean()
        # print(self.yearly_mean_values)
        self.yearly_mean_values.to_csv("./data/年均量.txt")

        stability = yearly_max_values / yearly_min_values

        second_result = df_filtered.groupby(df_filtered.index.month).apply(lambda x: (x > 60).sum())

        second_result_max = second_result.max()
        second_result_min = second_result.min()
        second_stability = second_result_max / second_result_min
        second_stability = second_stability.round(2)
        second_stability = pd.DataFrame(second_stability)
        second_stability.columns = ["stability"]

        data = second_stability.values
        # 表格加载数据
        # 设置行列，设置表头
        tmp = second_stability.columns
        tmp2 = [str(_) for _ in second_stability.index.tolist()]
        self.TableWidget.setRowCount(len(data))
        self.TableWidget.setColumnCount(len(data[0]))
        self.TableWidget.setHorizontalHeaderLabels(tmp)
        self.TableWidget.setVerticalHeaderLabels(tmp2)
        # 表格加载内容
        for row, form in enumerate(data):
            for column, item in enumerate(form):
                self.TableWidget.setItem(row, column, QTableWidgetItem(str(item)))

    #趋势分析
    def trend_analysis(self):

        time_series = self.yearly_mean_values.index
        lst = []
        for i in range(len(self.yearly_mean_values.columns)):
            runoff_data = self.yearly_mean_values.iloc[:,i].values
            slope, intercept, r_value, p_value, std_err = linear_regression(time_series, runoff_data)
            lst.append([slope.round(5), intercept.round(5),r_value.round(5), p_value.round(5), std_err.round(5)])
        trend_df = pd.DataFrame(lst,index=["slope", "intercept","r_value", "p_value", "std_err"],
                                columns=self.yearly_mean_values.columns)

        data = trend_df.values
        # 表格加载数据
        # 设置行列，设置表头
        tmp = trend_df.columns
        tmp2 = [str(_) for _ in trend_df.index.tolist()]
        self.TableWidget.setRowCount(len(data))
        self.TableWidget.setColumnCount(len(data[0]))
        self.TableWidget.setHorizontalHeaderLabels(tmp)
        self.TableWidget.setVerticalHeaderLabels(tmp2)
        # 表格加载内容
        for row, form in enumerate(data):
            for column, item in enumerate(form):
                if math.isnan(item):
                    self.TableWidget.setItem(row, column, QTableWidgetItem(""))
                else:
                    self.TableWidget.setItem(row, column, QTableWidgetItem(str(item)))

        # 保存数据
        path = QFileDialog.getSaveFileName(self, "保存文件", "./", ("结果(*.xlsx)"))
        if path:
            trend_df.to_excel(path[0])

    #突变分析
    def mutation_analysis(self):
        time_series = self.yearly_mean_values.index
        df_k = []
        df_lst = []
        for i in range(len(self.yearly_mean_values.columns)):
            year_runoff = self.yearly_mean_values.iloc[:,i].values
            k, UFk, UBkT = Kendall_change_point_detection(time_series, year_runoff)
            # print(k, UFk, UBkT)
            #返回三个列表
            df_k.append(int(time_series[item]) for item in k)
            df_lst.append(UFk)
            df_lst.append(UBkT)
        result_lst = pd.DataFrame(df_lst)
        result_lst = result_lst.T
        result_lst.index = self.yearly_mean_values.index
        columns_lst = []
        for item in self.yearly_mean_values.columns:
            columns_lst.append(f"{item}_UFk")
            columns_lst.append(f"{item}_UBk")
        result_lst.columns = columns_lst
        result_k = pd.DataFrame(df_k)
        result_k = result_k.T
        result_k.columns = self.yearly_mean_values.columns

        data = result_k.values
        # 表格加载数据
        # 设置行列，设置表头
        tmp = result_k.columns
        tmp2 = [str(_) for _ in result_k.index.tolist()]
        self.TableWidget.setRowCount(len(data))
        self.TableWidget.setColumnCount(len(data[0]))
        self.TableWidget.setHorizontalHeaderLabels(tmp)
        self.TableWidget.setVerticalHeaderLabels(tmp2)
        # 表格加载内容
        for row, form in enumerate(data):
            for column, item in enumerate(form):
                if math.isnan(item):
                    self.TableWidget.setItem(row, column, QTableWidgetItem(""))
                else:
                    self.TableWidget.setItem(row, column, QTableWidgetItem(str(int(item))))

        # 保存数据
        path = QFileDialog.getSaveFileName(self, "保存文件", "./", ("结果(*.xlsx)"))
        if path:
            # 创建Excel写入器对象
            writer = pd.ExcelWriter('output.xlsx')
            # 将第一个DataFrame保存为Sheet1
            result_lst.to_excel(writer, sheet_name='Ufk_ubk', index=False)
            # 将第二个DataFrame保存为Sheet2
            result_k.to_excel(writer, sheet_name='突变年份', index=False)
            # 关闭Excel写入器并保存文件
            writer.save()

    #年均趋势分析
    def avg_trend_analysis(self):

        time_series = self.df_avg.index
        lst = []
        for i in range(len(self.df_avg.columns)):
            runoff_data = self.df_avg.iloc[:,i].values
            slope, intercept, r_value, p_value, std_err = linear_regression(time_series, runoff_data)
            lst.append([slope.round(5), intercept.round(5),r_value.round(5), p_value.round(5), std_err.round(5)])
        trend_df = pd.DataFrame(lst,index=["slope", "intercept","r_value", "p_value", "std_err"],
                                columns=self.df_avg.columns)

        data = trend_df.values
        # 表格加载数据
        # 设置行列，设置表头
        tmp = trend_df.columns
        tmp2 = [str(_) for _ in trend_df.index.tolist()]
        self.TableWidget.setRowCount(len(data))
        self.TableWidget.setColumnCount(len(data[0]))
        self.TableWidget.setHorizontalHeaderLabels(tmp)
        self.TableWidget.setVerticalHeaderLabels(tmp2)
        # 表格加载内容
        for row, form in enumerate(data):
            for column, item in enumerate(form):
                if math.isnan(item):
                    self.TableWidget.setItem(row, column, QTableWidgetItem(""))
                else:
                    self.TableWidget.setItem(row, column, QTableWidgetItem(str(item)))

        # 保存数据
        path = QFileDialog.getSaveFileName(self, "保存文件", "./", ("结果(*.xlsx)"))
        if path:
            trend_df.to_excel(path[0])

    #年均突变分析
    def avg_mutation_analysis(self):
        time_series = self.df_avg.index
        df_k = []
        df_lst = []
        for i in range(len(self.df_avg.columns)):
            year_runoff = self.df_avg.iloc[:,i].values
            k, UFk, UBkT = Kendall_change_point_detection(time_series, year_runoff)
            # print(k, UFk, UBkT)
            #返回三个列表
            df_k.append(int(time_series[item]) for item in k)
            df_lst.append(UFk)
            df_lst.append(UBkT)
        result_lst = pd.DataFrame(df_lst)
        result_lst = result_lst.T
        result_lst.index = self.df_avg.index
        columns_lst = []
        for item in self.df_avg.columns:
            columns_lst.append(f"{item}_UFk")
            columns_lst.append(f"{item}_UBk")
        result_lst.columns = columns_lst
        result_k = pd.DataFrame(df_k)
        result_k = result_k.T
        result_k.columns = self.df_avg.columns

        data = result_k.values
        # 表格加载数据
        # 设置行列，设置表头
        tmp = result_k.columns
        tmp2 = [str(_) for _ in result_k.index.tolist()]
        self.TableWidget.setRowCount(len(data))
        self.TableWidget.setColumnCount(len(data[0]))
        self.TableWidget.setHorizontalHeaderLabels(tmp)
        self.TableWidget.setVerticalHeaderLabels(tmp2)
        # 表格加载内容
        for row, form in enumerate(data):
            for column, item in enumerate(form):
                if math.isnan(item):
                    self.TableWidget.setItem(row, column, QTableWidgetItem(""))
                else:
                    self.TableWidget.setItem(row, column, QTableWidgetItem(str(int(item))))

        # 保存数据
        path = QFileDialog.getSaveFileName(self, "保存文件", "./", ("结果(*.xlsx)"))
        if path:
            # 创建Excel写入器对象
            writer = pd.ExcelWriter('output.xlsx')
            # 将第一个DataFrame保存为Sheet1
            result_lst.to_excel(writer, sheet_name='Ufk_ubk', index=False)
            # 将第二个DataFrame保存为Sheet2
            result_k.to_excel(writer, sheet_name='突变年份', index=False)
            # 关闭Excel写入器并保存文件
            writer.save()

    def showDialog(self):
        title = '弹窗警告'
        content = """数据输入索引无法转化为时间索引，请对时间输入进行手动调整"""
        # w = MessageDialog(title, content, self)   # Win10 style message box
        w = MessageBox(title, content, self)
        if w.exec():
            pass
        else:
            pass

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

