#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
2016.5.26
相関をとってみる

2016.4.29
低周波のデータをカットする
各データの時系列変化の積算をとり、任意の閾値でカット

2016.4.20
データを複数回に分けて保存する
でかすぎるデータを一気に保存すると強制終了

2016.4.5
データの保存形式を変更する
見やすくする+メモリ節約のため

2016.3.24
x, y, zを利用する

"""

import sys
import os.path
import math
import json
import time
from datetime import datetime
import h5py
import tqdm

#calc
import numpy as np
from numpy import linalg as NLA
import scipy as sp
from scipy import linalg as SLA
from scipy.spatial import distance as DIST

#GUI
from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import *
from PyQt4.QtGui  import *

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

#plots
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#ROS
import rospy


class Plot():
    def __init__(self, parent=None, width=5, height=5, dpi=100):
        #pl.ion()
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(parent)
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.rho_area = self.fig.add_subplot(211)
        self.rho_area.set_title("rho", fontsize=11)
        self.tmp_x_area = self.fig.add_subplot(223)
        self.tmp_x_area.set_title("user 1", fontsize=11)
        self.tmp_y_area = self.fig.add_subplot(224)
        self.tmp_y_area.set_title("user 2", fontsize=11)
        self.fig.tight_layout()        

    #プロットエリアがクリックされた時
    def on_click(self, event):
        print 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(
            event.button, event.x, event.y, event.xdata, event.ydata)
        row = int(event.ydata)
        col = int(event.xdata)
        print "---(",row,", ",col,")---"
        print 'cca r:',self.r_m[row][col]

        """
        #データの可視化
        #print "x:",len(self.tmp_x[row][col]), len(self.tmp_x[row][col][0]), ", y:",len(self.tmp_y[row][col]),len(self.tmp_y[row][col][0])
        print "u1[0]:",self.tmp_x[row][col][0],", u2[0]:",self.tmp_y[row][col][0]

        #data1 = np.fabs(self.tmp_x[row][col]).mean(axis=1)
        #data2 = np.fabs(self.tmp_y[row][col]).mean(axis=1)

        self.tmp_x_area.cla()
        self.tmp_x_area.plot(self.tmp_x[row][col])
        #self.tmp_x_area.plot(data1)
        self.tmp_x_area.set_title("user 1", fontsize=11)
        self.tmp_x_area.set_ylim(-0.6, 0.6)

        self.tmp_y_area.cla()
        self.tmp_y_area.plot(self.tmp_y[row][col])
        #self.tmp_y_area.plot(data2)
        self.tmp_y_area.set_title("user 2", fontsize=11)
        self.tmp_y_area.set_ylim(-0.6, 0.6)
        self.fig.canvas.draw()
        self.fig.tight_layout()   
        """


        """
        #多重共線性をチェックするためvifを可視化
        tx = np.array(self.tmp_x[row][col])
        p1 = np.corrcoef(tx.T)
        ty = np.array(self.tmp_y[row][col])
        p2 = np.corrcoef(ty.T)
        
        print "tx:",tx.shape, ", ty:",ty.shape
        
        # ここで, tx, tyのshapeが(1,)とかになってると計算不能?
        print "data1 vif:"
        for r in range(p1.shape[0] - 1): #p1.shape[0]
            for c in range(r, p1.shape[1] -1):
                p = p1[r][c+1]
                vif = 1/(1-p**2)
                if vif > 10:
                    print r, c, vif

        print "data2 vif:"
        for r in range(p2.shape[0] - 1): 
            for c in range(r, p2.shape[1] -1):
                p = p2[r][c+1]
                vif = 1/(1-p**2)
                if vif > 10:
                    print r, c, vif

        """

    def scale(self, X):
        data = (X - np.mean(X, axis=0))/np.std(X, axis=0)
        return data

    def on_draw(self, r):
        self.r_m = r
        self.rho_area = self.fig.add_subplot(211)

        #データの可視化
        """
        self.tmp_x = x
        self.tmp_y = y
        self.tmp_x_area = self.fig.add_subplot(223)
        self.tmp_y_area = self.fig.add_subplot(224)
        """

        fs = 10
        dr, dc = self.r_m.shape
        Y, X = np.mgrid[slice(0, dr+1, 1),slice(0, dc+1, 1)]

        #img = self.rho_area.pcolor(X, Y, self.r_m, vmin=-1.0, vmax=1.0, cmap=cm.gray)#gray #bwr
        img = self.rho_area.pcolor(X, Y, self.r_m, vmin=-1.0, vmax=1.0, cmap=cm.bwr)#gray #bwr
        """
        if self.cbar == None:
            self.cbar = self.fig.colorbar(img)
            self.cbar.ax.tick_params(labelsize=fs-1) 
        """
        self.rho_area.set_xlim(0, dc)
        self.rho_area.set_ylim(0, dr)

        wid = 10 #とりあえず決め打ちで10ずつ目盛表示
        ticks = [i*wid for i in range(dr/wid+1)]
        labels = [(dr-1)/2-i*wid for i in range(dr/wid+1)]
        self.rho_area.set_yticks(ticks=ticks)
        self.rho_area.set_yticklabels(labels=labels)
        self.rho_area.set_xlabel("user 1")
        self.rho_area.set_ylabel("user 2")

        self.rho_area.tick_params(labelsize=fs)
        self.rho_area.set_title("rho", fontsize=fs+1)
        self.fig.canvas.draw()


class CCA(QtGui.QWidget):

    def __init__(self):
        super(CCA, self).__init__()
        #UI
        self.init_ui()
        #ROS
        rospy.init_node('pose_corr', anonymous=True)

    def init_ui(self):
        grid = QtGui.QGridLayout()
        form = QtGui.QFormLayout()
        
        #data file input box
        self.txtSepFile = QtGui.QLineEdit()
        btnSepFile = QtGui.QPushButton('...')
        btnSepFile.setMaximumWidth(40)
        btnSepFile.clicked.connect(self.choose_db_file)
        boxSepFile = QtGui.QHBoxLayout()
        boxSepFile.addWidget(self.txtSepFile)
        boxSepFile.addWidget(btnSepFile)
        form.addRow('input file', boxSepFile)

        #window size
        self.dataStart = QtGui.QLineEdit()
        self.dataStart.setText('0')
        self.dataStart.setFixedWidth(70)
        self.dataEnd = QtGui.QLineEdit()
        self.dataEnd.setText('500')
        self.dataEnd.setFixedWidth(70)

        boxDatas = QtGui.QHBoxLayout()
        boxDatas.addWidget(self.dataStart)
        boxDatas.addWidget(self.dataEnd)
        form.addRow('data range', boxDatas)

        #window size
        self.winSizeBox = QtGui.QLineEdit()
        self.winSizeBox.setText('34')
        self.winSizeBox.setAlignment(QtCore.Qt.AlignRight)
        self.winSizeBox.setFixedWidth(100)
        form.addRow('window size', self.winSizeBox)

        #frame size
        self.frmSizeBox = QtGui.QLineEdit()
        self.frmSizeBox.setText('68')
        self.frmSizeBox.setAlignment(QtCore.Qt.AlignRight)
        self.frmSizeBox.setFixedWidth(100)
        form.addRow('offset frames', self.frmSizeBox)

        # regulation
        self.regBox = QtGui.QLineEdit()
        self.regBox.setText('0.0')
        self.regBox.setAlignment(QtCore.Qt.AlignRight)
        self.regBox.setFixedWidth(100)
        form.addRow('regulation', self.regBox)

        # threshold
        self.thBox = QtGui.QLineEdit()
        self.thBox.setText('0.05')
        self.thBox.setAlignment(QtCore.Qt.AlignRight)
        self.thBox.setFixedWidth(100)
        form.addRow('threshold', self.thBox)

        rHLayout = QtGui.QHBoxLayout()
        self.radios = QtGui.QButtonGroup()
        self.allSlt = QtGui.QRadioButton('all frame')
        self.radios.addButton(self.allSlt)
        rHLayout.addWidget(self.allSlt)
        form.addRow('select', rHLayout)
 
        """
        #progress bar
        self.pBar = QtGui.QProgressBar()
        form.addRow('progress', self.pBar)
        """

        #exec button
        boxCtrl = QtGui.QHBoxLayout()
        btnExec = QtGui.QPushButton('exec')
        btnExec.clicked.connect(self.do_exec)
        #btnExec.clicked.connect(self.manyFileExec)
        boxCtrl.addWidget(btnExec)

        #output file
        boxPlot = QtGui.QHBoxLayout()
        btnPlot = QtGui.QPushButton('plot')
        btnPlot.clicked.connect(self.rhoplot)
        boxPlot.addWidget(btnPlot)

        #output file
        boxFile = QtGui.QHBoxLayout()
        btnOutput = QtGui.QPushButton('output')
        btnOutput.clicked.connect(self.save_params)
        boxFile.addWidget(btnOutput)

        # matplotlib
        boxPlot = QtGui.QHBoxLayout()
        self.main_frame = QtGui.QWidget()
        self.plot = Plot(self.main_frame)
        boxPlot.addWidget(self.plot.canvas)

        #配置
        grid.addLayout(form,1,0)
        grid.addLayout(boxCtrl,2,0)
        grid.addLayout(boxFile,3,0)
        grid.addLayout(boxPlot,4,0)

        self.setLayout(grid)
        #self.resize(400,100)

        self.setWindowTitle("cca window")
        self.show()

    def choose_db_file(self):
        dialog = QtGui.QFileDialog()
        dialog.setFileMode(QtGui.QFileDialog.ExistingFile)
        if dialog.exec_():
            fileNames = dialog.selectedFiles()
            for f in fileNames:
                self.txtSepFile.setText(f)
                return
        return self.txtSepFile.setText('')

    def updateColorTable(self, cItem):
        self.r = cItem.row()
        self.c = cItem.column()
        print "now viz r:",self.r,", c:",self.c

    def do_exec(self):
        print "exec start:",datetime.now().strftime("%Y/%m/%d %H:%M:%S")

        # input file
        self.fname = str(self.txtSepFile.text())
        self.data1, self.data2, self.dts, self.dtd = self.json_pose_input(self.fname)
        self.times = self.json_time_input(self.fname)
        # select joints
        self.data1, self.data2 = self.select_datas(self.data1, self.data2)
        # if data is big then...
        start, end = int(self.dataStart.text()), int(self.dataEnd.text())
        self.data1, self.data2, self.start, self.end = self.cut_datas(self.data1, self.data2, start, end)
        # data size update
        self.pre_dtd = self.dtd
        self.dts, self.dtd = self.data1.shape
        # data normalization
        #self.data1, self.data2, self.org1, self.org2 = self.normalize_datas(self.data1, self.data2)
        self.data1, self.org1 = self.normalize_data(self.data1)
        self.data2, self.org2 = self.normalize_data(self.data2)

        #ws:window_size, fs:frame_size 
        self.wins = int(self.winSizeBox.text())
        self.frms = int(self.frmSizeBox.text())
        if self.allSlt.isChecked() == True:
            print "frame all"
            self.frms = self.dts

        #dtmr:data_max_range,
        self.dtmr = self.dts - self.wins + 1

        print "datas_size:",self.dts
        print "frame_size:",self.frms
        print "data_max_range:",self.dtmr

        #rho_m:rho_matrix[dmr, dmr, datadimen] is corrs
        #wx_m and wy_m is vectors
        self.reg = float(self.regBox.text())
        self.th = float(self.thBox.text())
        self.r_m, self.wx_m, self.wy_m, self.js1, self.js2 = self.cca_exec(self.data1, self.data2)

        #plt.plot(self.data2)
        #plt.show()
        #self.r_m, self.tmp_x, self.tmp_y = self.cca_exec(self.data1, self.data2)

        #graph
        self.rhoplot()

        print "end:",datetime.now().strftime("%Y/%m/%d %H:%M:%S")

    def rhoplot(self):
        #self.plot.on_draw(self.r_m[:,:,0])
        self.plot.on_draw(self.r_m[:,:])
        
    def json_input(self, filename):
        f = open(filename, 'r')
        jD = json.load(f)
        f.close()
        #angle
        ds, dd = len(jD[0]["datas"]), len(jD[0]["datas"][0]["data"])
        datas = [[u["datas"][j]["data"] for j in range(ds)] for u in jD]
        return np.array(datas[0]), np.array(datas[1]), ds, dd

    def json_pose_input(self, filename):
        f = open(filename, 'r');
        jD = json.load(f)
        f.close()

        datas = []
        ds = len(jD[0]["datas"])
        dd = len(jD[0]["datas"][0]["jdata"]) 
        dp = len(jD[0]["datas"][0]["jdata"][0])

        for user in jD:
            pobj = []
            for s in range(ds):
                pl = []
                for d in range(dd):
                    for p in range(dp):
                        pl.append(user["datas"][s]["jdata"][d][p])
                pobj.append(pl)
            datas.append(pobj)

        return np.array(datas[0]), np.array(datas[1]), ds, dd*dp

    def json_time_input(self, filename):
        f = open(filename, 'r')
        jD = json.load(f)
        f.close()
        times = []
        if jD[0]["datas"][0].has_key("time"): 
            for tobj in jD[0]["datas"]:
                times.append(tobj["time"])
        else:
            print "[WARN] no time data!"
        return times        

    def save_params(self):
        save_dimen=1 #self.dtd
        savefile = "save_"+ self.fname.lstrip("/home/uema/catkin_ws/src/pose_cca/datas/")
        savefile = savefile.rstrip(".json")+"_w"+str(self.wins)+"_f"+str(self.frms) +"_d"+str(self.dtd)+"_r"+str(self.reg)+"_t"+str(self.th)+"_s"+str(self.start)+"_e"+str(self.end)+"_corr" 
        filepath = savefile+".h5"
        print filepath+" is save"
        with h5py.File(filepath, 'w') as f:
            p_grp=f.create_group("prop")
            p_grp.create_dataset("wins",data=self.wins)
            p_grp.create_dataset("frms",data=self.frms)
            p_grp.create_dataset("dtd",data=self.dtd)
            p_grp.create_dataset("pre_dtd",data=self.pre_dtd)
            p_grp.create_dataset("dts",data=self.dts) 
            p_grp.create_dataset("fname",data=self.fname)
            p_grp.create_dataset("sidx",data=self.sIdx)
            p_grp.create_dataset("reg",data=self.reg)
            p_grp.create_dataset("th",data=self.th)
            p_grp.create_dataset("org1",data=self.org1)
            p_grp.create_dataset("org2",data=self.org2)
            c_grp=f.create_group("cca")
            c_grp.create_dataset("times", data=self.times)
            d_grp=c_grp.create_group("data")
            d_grp.create_dataset("data1", data=self.data1)
            d_grp.create_dataset("data2", data=self.data2)
            r_grp=c_grp.create_group("r")
            wx_grp=c_grp.create_group("wx")
            wy_grp=c_grp.create_group("wy")

            #print "now save only r_m"
            #r_grp.create_dataset(str(0),data=self.r_m[:,:])
            """
            for i in xrange(save_dimen):
                r_grp.create_dataset(str(i),data=self.r_m[:,:,i])
                wx_v_grp = wx_grp.create_group(str(i))
                wy_v_grp = wy_grp.create_group(str(i))
                for j in xrange(self.dtd):
                    wx_v_grp.create_dataset(str(j),data=self.wx_m[:,:,j,i])
                    wy_v_grp.create_dataset(str(j),data=self.wy_m[:,:,j,i])
            """
            """
            # jointsの保存,ぶっちゃけいらないかも
            print self.js1
            print self.js2
            js_grp=c_grp.create_group("js")
            js_grp.create_dataset(str(0), data=self.js1)
            js_grp.create_dataset(str(1), data=self.js2)
            """
            #save_dimen = 1
            for i in xrange(save_dimen):
                r_grp.create_dataset(str(i),data=self.r_m[:,:])
                wx_v_grp = wx_grp.create_group(str(i))
                wy_v_grp = wy_grp.create_group(str(i))
                for j in xrange(self.dtd):
                    #print len(self.wx_m)
                    #print self.wx_m
                    wx_v_grp.create_dataset(str(j),data=self.wx_m[:,:,j])
                    wy_v_grp.create_dataset(str(j),data=self.wy_m[:,:,j])
            f.flush()
        print "save end:",datetime.now().strftime("%Y/%m/%d %H:%M:%S")

    def save_params2(self):
        save_dimen=1 #self.dtd
        savefile = "save_"+ self.fname.lstrip("/home/uema/catkin_ws/src/pose_cca/datas/")
        savefile = savefile.rstrip(".json")+"_w"+str(self.wins)+"_f"+str(self.frms) +"_d"+str(self.dtd)+"_r"+str(self.reg)+"_s"+str(self.start)+"_e"+str(self.end) 
        filepath = savefile+".h5"
        print filepath+" is save"
        with h5py.File(filepath, 'w') as f:
            p_grp=f.create_group("prop")
            p_grp.create_dataset("wins",data=self.wins)
            p_grp.create_dataset("frms",data=self.frms)
            p_grp.create_dataset("dtd",data=self.dtd)
            p_grp.create_dataset("pre_dtd",data=self.pre_dtd)
            p_grp.create_dataset("dts",data=self.dts) 
            p_grp.create_dataset("fname",data=self.fname)
            p_grp.create_dataset("sidx",data=self.sIdx)
            p_grp.create_dataset("reg",data=self.reg)
            p_grp.create_dataset("th",data=self.th)
            p_grp.create_dataset("org1",data=self.org1)
            p_grp.create_dataset("org2",data=self.org2)

            c_grp=f.create_group("cca")
            c_grp.create_dataset("times", data=self.times)

            # 生データ保存
            d_grp=c_grp.create_group("data")
            d_grp.create_dataset("data1", data=self.data1)
            d_grp.create_dataset("data2", data=self.data2)

            # 処理結果の保存
            rslt_dts = self.r_m.shape[1]
            rng = 1000
            p_grp.create_dataset("rng",data=rng)

            sp = 0
            while True:
                print "sp:",sp
                if (sp+1)*rng < rslt_dts:
                    print "rslt_dts:", rslt_dts, "rng*(sp+1)",rng*(sp+1)
                    sp_grp = c_grp.create_group(str(sp*rng)+"-"+str((sp+1)*rng))
                    r_grp = sp_grp.create_group("r")
                    wx_grp = sp_grp.create_group("wx")
                    wy_grp = sp_grp.create_group("wy")
                    for i in tqdm.tqdm(xrange(save_dimen)):
                        r_grp.create_dataset(str(i),data=self.r_m[:,sp*rng:(sp+1)*rng,i])
                        wx_v_grp = wx_grp.create_group(str(i))
                        wy_v_grp = wy_grp.create_group(str(i))
                        for j in tqdm.tqdm(xrange(self.dtd)):
                            wx_v_grp.create_dataset(str(j),data=self.wx_m[:,sp*rng:(sp+1)*rng,j,i])
                            wy_v_grp.create_dataset(str(j),data=self.wy_m[:,sp*rng:(sp+1)*rng,j,i])
                            f.flush()
                else:
                    sp_grp = c_grp.create_group(str(sp*rng)+"-"+str(rslt_dts-1))
                    r_grp = sp_grp.create_group("r")
                    wx_grp = sp_grp.create_group("wx")
                    wy_grp = sp_grp.create_group("wy")
                    for i in tqdm.tqdm(xrange(save_dimen)):
                        r_grp.create_dataset(str(i),data=self.r_m[:,sp*rng:rslt_dts-1,i])
                        wx_v_grp = wx_grp.create_group(str(i))
                        wy_v_grp = wy_grp.create_group(str(i))
                        for j in tqdm.tqdm(xrange(self.dtd)):
                            wx_v_grp.create_dataset(str(j),data=self.wx_m[:,sp*rng:rslt_dts-1,j,i])
                            wy_v_grp.create_dataset(str(j),data=self.wy_m[:,sp*rng:rslt_dts-1,j,i])
                            f.flush()
                    break
                sp += 1
            f.flush()
            f.close()
        print "save end:",datetime.now().strftime("%Y/%m/%d %H:%M:%S")


    def id_conv(self, idx):
        return [idx*3+i for i in range(3)]



    def normalize_data(self, data):
        datas = []
        os = 3
        org = self.std_data(data[0], os)[1]
        for dt in data:
            datas.append(self.std_data(dt, os)[0])
        return np.array(datas), org


    def std_data(self, data, os):
        # 原点の計算
        def calc_org(data, s_l_id, s_r_id, spn_id, os):
            #print data
            s_l, s_r, spn = data[s_l_id], data[s_r_id], data[spn_id] 
            a, b = 0, 0
            # 原点 = 右肩から左肩にかけた辺と胸からの垂線の交点
            for i in range(os):
                a += (spn[i]-s_l[i])*(s_r[i]-s_l[i])
                b += (s_r[i]-s_l[i])**2
            k = a/b
            return np.array([k*s_r[i]+(1-k)*s_l[i] for i in range(os)])

        # 法線の計算
        def calc_normal(d_sl, d_sp, d_sr):
            l_s = np.array(d_sl) - np.array(d_sp)
            s_r = np.array(d_sp) - np.array(d_sr)
            x = l_s[1]*s_r[2]-l_s[2]*s_r[1]
            y = l_s[2]*s_r[0]-l_s[0]*s_r[2]
            z = l_s[0]*s_r[1]-l_s[1]*s_r[0]
            return np.array([x, y, z])

        # 回転行列による変換
        def calc_rot_pose(data, th_z, th_y, org):
            cos_th_z, sin_th_z = np.cos(-th_z), np.sin(-th_z) 
            cos_th_y, sin_th_y = np.cos(th_y), np.sin(th_y)
            rot1 = np.array([[cos_th_z, -sin_th_z, 0],[sin_th_z, cos_th_z, 0],[0, 0, 1]])
            rot2 = np.array([[cos_th_y, 0, sin_th_y],[0, 1, 0],[-sin_th_y, 0, cos_th_y]])
            rot_pose = []
            for dt in data:
                dt = np.array(dt)-org
                rd = np.dot(rot1, dt)
                rd = np.dot(rot2, rd)
                rd = rd+org
                rot_pose.append(rd)
            return rot_pose

        # 平行移動
        def calc_shift_pose(data, org, s):
            shift = s - org
            shift_pose = []
            for dt in data:
                dt += shift
                shift_pose.append(dt)
            return shift_pose

        def data_set(data, os):
            ds = np.reshape(data, (len(data)/os, os))
            return ds

        def data_reset(data):
            data = np.array(data)
            ds = np.reshape(data, (data.shape[0]*data.shape[1], ))
            return ds
    

        #データを一旦xyzを配列のパックに戻す
        data = data_set(data, os)

        # 左肩 4, 右肩 8(カットせずに数えた場合のjoint index), 胸 1 
        s_l_id, s_r_id, spn_id = 4, 7, 1

        # 原点の計算
        org = calc_org(data, s_l_id, s_r_id, spn_id, os)
        # 法線を求める
        normal = calc_normal(data[s_l_id], data[spn_id], data[s_r_id])
        # 法線の角度方向にposeを回転
        th_z = np.arctan2(normal[1], normal[0])-np.arctan2(org[1], org[0]) #z軸回転(法線と原点の間の角度)
        th_y = np.arctan2(normal[2], normal[0])-np.arctan2(org[2], org[0]) #y軸回転 
        rot_pose = calc_rot_pose(data, th_z, th_y, org)
        #orgをx軸上に変換する
        th_z = np.arctan2(org[1], org[0])
        th_y = np.arctan2(org[2], org[0])
        rot_pose_norm = calc_rot_pose(rot_pose, th_z, th_y, np.array([1,0,0]))
        # 変換後のorg
        rot_org = calc_org(rot_pose_norm, s_l_id, s_r_id, spn_id, os)
        # orgのxを特定の値に移動する
        s = [0,0,0]
        shift_pose = calc_shift_pose(rot_pose_norm, rot_org, s)
        shift_org = calc_org(shift_pose, s_l_id, s_r_id, spn_id, os)

        pose = data_reset(shift_pose)
        return pose, shift_org 







    """
    def std_data(self, data, os):
        # 原点の計算
        def calc_org(data, s_l_id, s_r_id, spn_id, os):
            s_l, s_r, spn = data[s_l_id:s_l_id+os], data[s_r_id:s_r_id+os], data[spn_id:spn_id+os] 
            a, b = 0, 0
            # 原点 = 右肩から左肩にかけた辺と胸からの垂線の交点
            for i in range(os):
                a += (spn[i]-s_l[i])*(s_r[i]-s_l[i])
                b += (s_r[i]-s_l[i])**2
            k = a/b
            return np.array([k*s_r[i]+(1-k)*s_l[i] for i in range(os)])

        # 法線の計算
        def calc_normal(d_sl, d_sp, d_sr):
            l_s = np.array(d_sl) - np.array(d_sp)
            s_r = np.array(d_sp) - np.array(d_sr)
            x = l_s[1]*s_r[2]-l_s[2]*s_r[1]
            y = l_s[2]*s_r[0]-l_s[0]*s_r[2]
            z = l_s[0]*s_r[1]-l_s[1]*s_r[0]
            return np.array([x, y, z])

        # 回転行列による変換
        def calc_rot_pose(data, th_z, th_y, org):
            cos_th_z, sin_th_z = np.cos(-th_z), np.sin(-th_z) 
            cos_th_y, sin_th_y = np.cos(th_y), np.sin(th_y)
            rot1 = np.array([[cos_th_z, -sin_th_z, 0],[sin_th_z, cos_th_z, 0],[0, 0, 1]])
            rot2 = np.array([[cos_th_y, 0, sin_th_y],[0, 1, 0],[-sin_th_y, 0, cos_th_y]])
            rot_pose = []
            #ベクトルじゃない！3ずつ(xyz)だからそれ用に書き換える!!
            for dt in data:
                dt = np.array(dt)-org
                rd = np.dot(rot1, dt)
                rd = np.dot(rot2, rd)
                rd = rd+org
                rot_pose.append(rd)
            return rot_pose

        # 平行移動
        def calc_shift_pose(data, org, s):
            shift = s - org
            shift_pose = []
            print org.shape
            print data.shape
            for dt in data:
                dt += shift
                shift_pose.append(dt)
            return shift_pose

        def data_set(data, os):

            ds = np.reshape(data, (len(data)/os, os))
            return ds

        def data_reset(data):
            data = np.array(data)
            ds = np.reshape(data, (data.shape[0]*data.shape[1], ))
            return ds
    

        #データを一旦xyzを配列のパックに戻す
        data = data_set(data, os)
        # 左肩4, 右肩7(カットせずに数えた場合のjoint indexは8), 胸1
        s_l_id, s_r_id, spn_id = 4, 7, 1
        # 原点の計算
        org = calc_org(data, s_l_id, s_r_id, spn_id, os)
        # 法線を求める
        normal = calc_normal(data[s_l_id:s_l_id+os], data[spn_id:spn_id+os], data[s_r_id:s_r_id+os])
        # 法線の角度方向にposeを回転
        th_z = np.arctan2(normal[1], normal[0])-np.arctan2(org[1], org[0]) #z軸回転(法線と原点の間の角度)
        th_y = np.arctan2(normal[2], normal[0])-np.arctan2(org[2], org[0]) #y軸回転 
        rot_pose = calc_rot_pose(data, th_z, th_y, org)
        #orgをx軸上に変換する
        th_z = np.arctan2(org[1], org[0])
        th_y = np.arctan2(org[2], org[0])
        rot_pose_norm = calc_rot_pose(rot_pose, th_z, th_y, np.array([0,0,0]))
        # 変換後のorg
        rot_org = calc_org(rot_pose_norm, s_l_id, s_r_id, spn_id, os)
        # orgのxを特定の値に移動する
        s = [1,0,0]
        shift_pose = calc_shift_pose(rot_pose_norm, rot_org, s)
        shift_org = calc_org(shift_pose, s_l_id, s_r_id, spn_id, os)

        #データの形を(51,)に戻す
        pose = data_reset(shift_pose)

        return pose, shift_org 
    """


    def select_datas(self, data1, data2):
        self.sIdx = [0,1,2,3,4,5,6,8,9,10,12,13,14,16,17,18,20]
        new_sid = [sid*3+i  for sid in self.sIdx for i in xrange(3)]
        self.sIdx = new_sid
        #print self.sIdx
        return data1[:,self.sIdx],  data2[:,self.sIdx]

    def cut_datas(self, data1, data2, start, end): 
        #print "s/e",start, end
        if start <= end and end <= self.dts:
            return data1[start:end,:], data2[start:end,:], start, end
        else:
            print "no cut"
            return data1, data2, 0, self.dts
    
    def low_var_cut(self, X, th): 
        Xp = []
        for i, xdata in enumerate(np.array(X).T):
            Xvar = np.var(np.array(xdata))
            if Xvar >= th:
                Xp.append(xdata)
                #print "x",i, Xvar
        Xc = np.array(Xp).T
        return Xc

    def low_std_cut_joints(self, X, th):
        exts = []
        std_tmps = []

        for i, xdata in enumerate(np.array(X).T):
            Xstd = np.std(np.array(xdata))
            std_tmps.append(Xstd)
            if Xstd >= th:
                exts.append(i)
                #print i
                #print xdata 
                #print Xstd

        #print "joints len:",len(std_tmps)
        # もしひとつも閾値thを超えなければ,最大を返す
        """
        if len(exts) == 0:
            exts.append(np.array(std_tmps).argmax())
        """
        return exts
        

    def extract(self, X, th):
        exts = []
        L = X.shape[0]
        W = X.shape[1] 

        # 何個とばしにするかで、みる動きが異なる
        for d in range(W):
            sum = 0
            for n in range(L-1):
                sum += np.fabs(X[n+1, d] - X[n, d])           
            if sum > th:
                #print d, sum
                exts.append(X[:,d])

        return np.array(exts).T

    # 使用するjointのインデックスを返す(配列)
    def ex_joints(self, X, th):
        exts = []
        L = X.shape[0]
        W = X.shape[1] 
        for d in range(W):
            sum = 0
            for n in range(L-1):
                sum += np.fabs(X[n+1, d] - X[n, d])           
            if sum > th:
                exts.append(d)
        return exts


    def cca_exec(self, data1, data2):
        #rho_m:rho_matrix[dmr, dmr, datadimen] is corrs
        #wx_m and wy_m is vectors
        data1 = np.array(data1)
        data2 = np.array(data2)

        r_m = np.zeros([self.frms*2+1, self.dtmr])
        #r_m = np.zeros([self.frms*2+1, self.dtmr, self.dtd])
        wx_m = np.zeros([self.frms*2+1, self.dtmr, self.dtd])
        wy_m = np.zeros([self.frms*2+1, self.dtmr, self.dtd])

        # 使われたデータを可視化したいのでそのためのバッファ
        #tmp_x = np.zeros([self.frms*2+1, self.dtmr, self.dtd, self.wins])
        #tmp_y = np.zeros([self.frms*2+1, self.dtmr, self.dtd, self.wins])
        tmp_x =[[[[]]]*self.dtmr]*(self.frms*2+1)
        tmp_y =[[[[]]]*self.dtmr]*(self.frms*2+1)

        js1 = [[[]]*self.dtmr]*(self.frms*2+1)
        js2 = [[[]]*self.dtmr]*(self.frms*2+1)

        #th = 0.3
        #row->colの順番で回したほうが効率いい

        #plt.ion()
        for i in tqdm.tqdm(xrange(self.dtmr)):
            for j in xrange(self.frms*2+1):
                if self.frms+i-j >= 0 and self.frms+i-j < self.dtmr:
                    u1 = data1[i:i+self.wins, :]                
                    u2 = data2[self.frms+i-j:self.frms+i-j+self.wins,:]

                    j1, j2 = self.low_std_cut_joints(u1, self.th), self.low_std_cut_joints(u2, self.th)
                    #j1, j2 = self.ex_joints(u1, self.th), self.ex_joints(u2, self.th)
                    #u1, u2 = self.low_var_cut(u1, 0.001), self.low_var_cut(u2, 0.001)
                    
                    #print u1[:,j1]
                    if len(j1) < 1 or len(j2) < 1: 
                        r_m[j][i] = 0
                    else:
                        #r_m[j][i], wx_m[j][i], wy_m[j][i] = self.cca(u1, u2, j1, j2)
                        r_m[j][i] = self.corr(u1, u2, j1, j2)
                        js1[j][i], js2[j][i] = json.dumps(j1), json.dumps(j2)

        return r_m, wx_m, wy_m, js1, js2

    def dset(self, X):
        tmps = []
        for d in X:
            tmp = []
            for v in d:
                tmp.append(v)
            tmps.append(tmp)
        return tmps

    def corr(self, X, Y, j1, j2):
        X = X[:, j1]
        Y = Y[:, j2]
        #print X.shape, Y.shape # ex:(34, 5)
        data1 = np.fabs(X).mean(axis=1)
        data2 = np.fabs(Y).mean(axis=1)

        return np.corrcoef(data1.T, data2.T)[0][1]

    def cca(self, X, Y, j1, j2):
        # ref: https://gist.github.com/satomacoto/5329310
        '''
        正準相関分析
        http://en.wikipedia.org/wiki/Canonical_correlation
        '''    
        X = X[:, j1]
        Y = Y[:, j2]

        n, p = X.shape
        n, q = Y.shape
                  
        # zero mean
        X = X - X.mean(axis=0)
        Y = Y - Y.mean(axis=0)
        
        # covariances
        S = np.cov(X.T, Y.T, bias=1)
        
        SXX = S[:p,:p]
        SYY = S[p:,p:]
        SXY = S[:p,p:]

        #正則化
        SXX = self.add_reg(SXX, self.reg) 
        SYY = self.add_reg(SYY, self.reg)

        sqx = SLA.sqrtm(SLA.inv(SXX)) # SXX^(-1/2)
        sqy = SLA.sqrtm(SLA.inv(SYY)) # SYY^(-1/2)
        M = np.dot(np.dot(sqx, SXY), sqy.T) # SXX^(-1/2) * SXY * SYY^(-T/2)
        A, r, Bh = SLA.svd(M, full_matrices=False)
        B = Bh.T     

        # 基底変換
        #A = np.dot(sqx, A)
        #B = np.dot(sqy, B)

        Wx = np.zeros([self.dtd])
        Wy = np.zeros([self.dtd])

        Wx[j1] = A[:,0]
        Wy[j2] = B[:,0]
        return r[0], Wx, Wy

    def add_reg(self, reg_cov, reg):
        reg_cov += reg * np.average(np.diag(reg_cov)) * np.identity(reg_cov.shape[0])
        return reg_cov

    """
    def gaussian_kernel(x, y, var=1.0):
        return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * var))

    def polynomial_kernel(x, y, c=1.0, d=2.0):
        return (np.dot(x, y) + c) ** d

    def kcca(self, X, Y, kernel_x=gaussian_kernel, kernel_y=gaussian_kernel, eta=1.0):
        n, p = X.shape
        n, q = Y.shape
        
        Kx = DIST.squareform(DIST.pdist(X, kernel_x))
        Ky = DIST.squareform(DIST.pdist(Y, kernel_y))
        J = np.eye(n) - np.ones((n, n)) / n
        M = np.dot(np.dot(Kx.T, J), Ky) / n
        L = np.dot(np.dot(Kx.T, J), Kx) / n + eta * Kx
        N = np.dot(np.dot(Ky.T, J), Ky) / n + eta * Ky


        sqx = SLA.sqrtm(SLA.inv(L))
        sqy = SLA.sqrtm(SLA.inv(N))
        
        a = np.dot(np.dot(sqx, M), sqy.T)
        A, s, Bh = SLA.svd(a, full_matrices=False)
        B = Bh.T
        
        # U = np.dot(np.dot(A.T, sqx), X).T
        # V = np.dot(np.dot(B.T, sqy), Y).T
        print s.shape
        print A.shape
        print B.shape
        return s, A, B
    """

def main():
    app = QtGui.QApplication(sys.argv)
    corr = CCA()
    sys.exit(app.exec_())

if __name__=='__main__':
    main()
