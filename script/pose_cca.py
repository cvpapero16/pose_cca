#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
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
        self.rho_area = self.fig.add_subplot(111)
        

    #プロットエリアがクリックされた時
    def on_click(self, event):
        print 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(
            event.button, event.x, event.y, event.xdata, event.ydata)
        row = int(event.ydata)
        col = int(event.xdata)
        print 'cca r:',self.r_m[row][col]

    def on_draw(self, r):
        self.r_m = r
        self.rho_area = self.fig.add_subplot(111)
        fs = 10
        dr, dc = self.r_m.shape
        Y, X = np.mgrid[slice(0, dr+1, 1),slice(0, dc+1, 1)]

        img = self.rho_area.pcolor(X, Y, self.r_m, vmin=0.0, vmax=1.0, cmap=cm.gray)
        """
        if self.cbar == None:
            self.cbar = self.fig.colorbar(img)
            self.cbar.ax.tick_params(labelsize=fs-1) 
        """
        self.rho_area.set_xlim(0, dc)
        self.rho_area.set_ylim(0, dr)
        self.rho_area.tick_params(labelsize=fs)
        self.rho_area.set_title("rho", fontsize=fs+1)
        self.fig.canvas.draw()


class CCA(QtGui.QWidget):

    def __init__(self):
        super(CCA, self).__init__()
        #UI
        self.init_ui()
        #ROS
        rospy.init_node('calccca', anonymous=True)


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
        """
        #ファイル出力
        self.txtSepFileOut = QtGui.QLineEdit()
        btnSepFileOut = QtGui.QPushButton('...')
        btnSepFileOut.setMaximumWidth(40)
        btnSepFileOut.clicked.connect(self.chooseOutFile)
        boxSepFileOut = QtGui.QHBoxLayout()
        boxSepFileOut.addWidget(self.txtSepFileOut)
        boxSepFileOut.addWidget(btnSepFileOut)
        form.addRow('output file', boxSepFileOut)    
        """
        #window size
        self.winSizeBox = QtGui.QLineEdit()
        self.winSizeBox.setText('90')
        self.winSizeBox.setAlignment(QtCore.Qt.AlignRight)
        self.winSizeBox.setFixedWidth(100)
        form.addRow('window size', self.winSizeBox)

        #frame size
        self.frmSizeBox = QtGui.QLineEdit()
        self.frmSizeBox.setText('110')
        self.frmSizeBox.setAlignment(QtCore.Qt.AlignRight)
        self.frmSizeBox.setFixedWidth(100)
        form.addRow('frame size', self.frmSizeBox)

        # regulation
        self.regBox = QtGui.QLineEdit()
        self.regBox.setText('0.01')
        self.regBox.setAlignment(QtCore.Qt.AlignRight)
        self.regBox.setFixedWidth(100)
        form.addRow('regulation', self.regBox)

        rHLayout = QtGui.QHBoxLayout()
        self.radios = QtGui.QButtonGroup()
        self.selected = QtGui.QRadioButton('dimension')
        self.radios.addButton(self.selected)
        self.allSlt = QtGui.QRadioButton('all frame')
        self.radios.addButton(self.allSlt)
        #form.addRow('dimension', self.selected)
        #form.addRow('select', self.allSlt)
        rHLayout.addWidget(self.selected)
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
        #grid.addLayout(boxPlot,3,0)
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
        #print "datas diff1",self.data1-self.data2
        #print self.data1.shape
        # select joints
        self.data1, self.data2 = self.select_datas(self.data1, self.data2)
        #print "datas diff2",self.data1-self.data2
        # if data is big then...
        #self.data1, self.data2 = self.cut_datas(self.data1, self.data2, 400)
        #print "datas diff3",self.data1-self.data2
        # data size update!
        self.pre_dtd = self.dtd
        self.dts, self.dtd = self.data1.shape
        # data normalization
        self.data1, self.data2, self.org1, self.org2 = self.normalize_datas(self.data1, self.data2)



        #ws:window_size, fs:frame_size 
        self.wins = int(self.winSizeBox.text())
        self.frms = int(self.frmSizeBox.text())
        if self.allSlt.isChecked() == True:
            print "frame all"
            self.frms = self.dts

        #dmr:data_max_range, frmr:frame_range, dtr:data_range
        self.dtmr = self.dts - self.wins + 1
        self.frmr = self.dts - self.frms + 1
        self.dtr = self.frms - self.wins + 1

        print "datas_size:",self.dts
        print "frame_size:",self.frms
        print "data_max_range:",self.dtmr
        print "frame_range:",self.frmr
        print "data_range:",self.dtr

        #rho_m:rho_matrix[dmr, dmr, datadimen] is corrs
        #wx_m and wy_m is vectors
        self.reg = float(self.regBox.text())
        self.r_m, self.wx_m, self.wy_m = self.cca_exec(self.data1, self.data2)

        #print self.r_m

        #graph
        self.rhoplot()

        print "end:",datetime.now().strftime("%Y/%m/%d %H:%M:%S")

    def rhoplot(self):
        self.plot.on_draw(self.r_m[:,:,0])
        
    def json_input(self, filename):
        f = open(filename, 'r')
        jD = json.load(f)
        f.close()
        #angle
        ds, dd = len(jD[0]["datas"]), len(jD[0]["datas"][0]["data"])
        datas = [[u["datas"][j]["data"] for j in range(ds)] for u in jD]
        return np.array(datas[0]), np.array(datas[1]), ds, dd

    """
    #この書き方はjointの順番がバラバラになる
    def json_pose_input(self, filename):
        f = open(filename, 'r')
        jD = json.load(f)
        f.close()
        #pose
        ds, dd, dp = len(jD[0]["datas"]), len(jD[0]["datas"][0]["jdata"]), len(jD[0]["datas"][0]["jdata"][0])  
        datas=[[[u["datas"][t]["jdata"][p][i]for i in range(dp) for p in range(dd)] for t in range(ds)] for u in jD]  
        return np.array(datas[0]), np.array(datas[1]), ds, dd*dp
    """
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


    def save_params(self):

        save_dimen=10 #self.dtd

        savefile = "save_w"+str(self.wins)+"_f"+str(self.frms) +"_d"+str(self.dtd)+"_r"+str(self.reg)+"_"+ self.fname.lstrip("/home/uema/catkin_ws/src/pose_cca/datas/") 
        savefile = savefile.rstrip(".json")
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
            p_grp.create_dataset("org1",data=self.org1)
            p_grp.create_dataset("org2",data=self.org2)
            c_grp=f.create_group("cca")
            r_grp=c_grp.create_group("r")
            wx_grp=c_grp.create_group("wx")
            wy_grp=c_grp.create_group("wy")
            d_grp=c_grp.create_group("data")
            d_grp.create_dataset("data1", data=self.data1)
            d_grp.create_dataset("data2", data=self.data2)
            for i in xrange(save_dimen):
                r_grp.create_dataset(str(i),data=self.r_m[:,:,i])
                wx_v_grp = wx_grp.create_group(str(i))
                wy_v_grp = wy_grp.create_group(str(i))
                for j in xrange(self.dtd):
                    wx_v_grp.create_dataset(str(j),data=self.wx_m[:,:,j,i])
                    wy_v_grp.create_dataset(str(j),data=self.wy_m[:,:,j,i])

            f.flush()
        print "save end:",datetime.now().strftime("%Y/%m/%d %H:%M:%S")


    def id_conv(self, idx):
        return [idx*3+i for i in range(3)]

    def normalize_datas(self, data1, data2):
       
        s_l_id, s_r_id, spn_id = 4, 7, 1
        offset = 3
        datas, orgs = [], []
        for dt in [data1, data2]:
            s_l = dt[0][s_l_id*offset:s_l_id*offset+offset] # 左肩 4
            s_r = dt[0][s_r_id*offset:s_r_id*offset+offset] # 右肩 7 
            spn = dt[0][spn_id*offset:spn_id*offset+offset] # 胸 1
            a, b = 0, 0
            for i in range(offset):
                a += (spn[i]-s_l[i])*(s_r[i]-s_l[i])
                b += (s_r[i]-s_l[i])**2
            k = a/b
            
            # 原点
            org = [k*s_r[i]+(1-k)*s_l[i] for i in range(offset)]
            orgs.append(org)
            print "org",org

            #x, y, zの塊ずつ見ていく
            data = []
            for d in dt:
                dtset = []
                for u in range(len(d)/3):
                    dtset.extend(d[u*3:u*3+3] - org)
                data.append(dtset)

            datas.append(data)

        print np.array(datas[0]).shape

        return datas[0], datas[1], orgs[0], orgs[1]

    def select_datas(self, data1, data2):
        self.sIdx = [0,1,2,3,4,5,6,8,9,10,12,13,14,16,17,18,20]
        new_sid = [sid*3+i  for sid in self.sIdx for i in xrange(3)]
        self.sIdx = new_sid
        print self.sIdx
        return data1[:,self.sIdx],  data2[:,self.sIdx]

    def cut_datas(self, data1, data2, th): 
        if self.dts < th:
            return data1, data2
        else:
            return data1[0:th,:], data2[0:th,:]

    def cca_exec(self, data1, data2):
        #rho_m:rho_matrix[dmr, dmr, datadimen] is corrs
        #wx_m and wy_m is vectors
        data1 = np.array(data1)
        data2 = np.array(data2)
        r_m = np.zeros([self.dtmr, self.dtmr, self.dtd])
        wx_m = np.zeros([self.dtmr, self.dtmr, self.dtd, self.dtd])
        wy_m = np.zeros([self.dtmr, self.dtmr, self.dtd, self.dtd])

        for f in tqdm.tqdm(xrange(self.frmr)):
            #self.pBar.setValue(f)
            if f == 0:
                for t1 in tqdm.tqdm(range(self.dtr)):
                    for t2 in xrange(self.dtr):
                        u1 = data1[f+t1:f+t1+self.wins,:]
                        u2 = data2[f+t2:f+t2+self.wins,:]
                        r_m[f+t1][f+t2], wx_m[f+t1][f+t2], wy_m[f+t1][f+t2] = self.cca(u1, u2)
            else:
                od = f+self.dtr-1
                for t1 in xrange(self.dtr-1):
                    u1 = data1[f+t1:f+t1+self.wins,:]
                    u2 = data2[od:od+self.wins,:]
                    r_m[f+t1][od], wx_m[f+t1][od], wy_m[f+t1][od] = self.cca(u1, u2)
                for t2 in xrange(self.dtr):
                    u1 = data1[od:od+self.wins,:]
                    u2 = data2[f+t2:f+t2+self.wins,:]
                    r_m[od][f+t2], wx_m[od][f+t2], wy_m[od][f+t2] = self.cca(u1, u2)

        return r_m, wx_m, wy_m


    def cca(self, X, Y):
        '''
        正準相関分析
        http://en.wikipedia.org/wiki/Canonical_correlation
        '''    
        #X = np.array(X)
        #Y = np.array(Y)
        #print X.shape
        n, p = X.shape
        n, q = Y.shape
        
        # zero mean
        X = X - X.mean(axis=0)
        Y = Y - Y.mean(axis=0)
        
        # covariances
        S = np.cov(X.T, Y.T, bias=1)
        
        # S = np.corrcoef(X.T, Y.T)
        SXX = S[:p,:p]
        SYY = S[p:,p:]
        SXY = S[:p,p:]
        #SYX = S[p:,:p]
        
        #正則化
        Rg = np.diag(np.ones(p)*self.reg)
        SXX = SXX + Rg
        SYY = SYY + Rg

        sqx = SLA.sqrtm(SLA.inv(SXX)) # SXX^(-1/2)
        sqy = SLA.sqrtm(SLA.inv(SYY)) # SYY^(-1/2)
        M = np.dot(np.dot(sqx, SXY), sqy.T) # SXX^(-1/2) * SXY * SYY^(-T/2)
        A, r, Bh = SLA.svd(M, full_matrices=False)
        B = Bh.T     
        #A = np.dot(SLA.sqrtm(SXX), A)
        #B = np.dot(SLA.sqrtm(SYY), Bh.T)
        #r = np.sqrt(r)
        return r, A, B


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



def main():
    app = QtGui.QApplication(sys.argv)
    corr = CCA()
    sys.exit(app.exec_())

if __name__=='__main__':
    main()
