#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
2016.5.17
pose_cca3の可視化

2016.4.27
正準ベクトルの可視化

2016.4.6
pose_cca2で作ったデータの可視化
基本的には変わらないが、rowとcolの意味合いが異なる

2016.3.25
pose, arrow


"""

import sys
import os.path
import math
import json
import time
import h5py

import numpy as np
from numpy import linalg as NLA
import scipy as sp
from scipy import linalg as SLA

from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import *
from PyQt4.QtGui  import *

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

import rospy
from visualization_msgs.msg import MarkerArray
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from geometry_msgs.msg import PointStamped
from std_msgs.msg import ColorRGBA


class Plot():
    def __init__(self, parent=None, width=6, height=7, dpi=100):
        self.fs = 8

        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(parent)
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.rho_area = self.fig.add_subplot(211)
        self.rho_area.tick_params(labelsize=self.fs)
        self.rho_area.set_title("rho",fontsize=self.fs+1)
        self.wx_area = self.fig.add_subplot(425)
        self.wx_area.tick_params(labelsize=self.fs)
        self.wx_area.set_title("user1 vec",fontsize=self.fs+1)
        self.wy_area = self.fig.add_subplot(427)
        self.wy_area.tick_params(labelsize=self.fs)
        self.wy_area.set_title("user2 vec",fontsize=self.fs+1)
        self.sig1_area = self.fig.add_subplot(426)
        self.sig1_area.tick_params(labelsize=self.fs)
        self.sig1_area.set_title("sig1:",fontsize=self.fs+1)

        self.sig2_area = self.fig.add_subplot(428)
        self.sig2_area.tick_params(labelsize=self.fs)
        self.sig2_area.set_title("sig2:",fontsize=self.fs+1)

        self.cbar = None
        self.fig.tight_layout()

    def on_draw(self, r, wx, wy, d1, d2, ws, si, pre_dtd):
        self.r_m = r
        self.wx_m = wx
        self.wy_m = wy
        self.data1 = d1
        self.data2 = d2
        self.wins = ws
        self.sidx = si
        self.pre_dtd = pre_dtd

        self.draw_rho()

    def staticCoef(self, d, v):        
        f = np.dot(d, v)

        #dの"重み"がない次元データは0にする
        js = self.get_index(v)#重みの入ってるインデックスのリスト
        
        #データが入ってるとこだけ計算する
        fu = np.corrcoef(np.c_[f,d[:,js]].T)[0,1:]
        #return f, fu
        #データを指定したindexにだけ入れなおす
        # よく考えたらそれって簡単にできないんじゃね?
        """
        wcf=[0]*51
        for i in range(len(self.sidx)):
            wcf[self.sidx[i]] = fu[i]
        """
        return fu, np.mean(fu**2)

    #プロットエリアがクリックされた時
    def on_click(self, event):
        col = int(event.xdata)
        row = int(event.ydata)
        print 'cca(%d, %d) r:%f'%(row, col, self.r_m[row][col])
        self.draw_weight(row, col)

        #dataの場合はズレを考慮に入れる
        width = (self.r_m.shape[0]-1)/2
        self.draw_sig(row, col, width)

        #構造係数
        #self.draw_static_coef(row, col, width)

        
        CCA().set_row_col(row, col)


    def draw_weight(self, row, col):

        dimen = self.pre_dtd
        xl = np.arange(dimen)

        self.wxlist=[0]*dimen
        for i in range(len(self.sidx)):
            self.wxlist[self.sidx[i]] = self.wx_m[i, row, col]
        
        self.wx_area.cla()
        self.wx_area.bar(xl, self.wxlist)
        self.wx_area.set_xlim(0, dimen)
        #self.wx_area.set_ylim(-1,1)
        #self.wx_area.tick_params(labelsize=self.fs)
        self.wx_area.set_title("user1 f: "+str(col),fontsize=self.fs+1)

        self.wylist=[0]*dimen
        for i in range(len(self.sidx)):
            self.wylist[self.sidx[i]] = self.wy_m[:,row,col][i]

        self.wy_area.cla()
        self.wy_area.bar(xl, self.wylist)
        self.wy_area.set_xlim(0, dimen)
        #self.wy_area.set_ylim(-1,1)
        #self.wy_area.tick_params(labelsize=self.fs)
        width = (self.r_m.shape[0]-1)/2
        self.wy_area.set_title("user2 f: "+str(width-row),fontsize=self.fs+1)

        self.fig.canvas.draw()


    def draw_static_coef(self, row, col, width):

        d1 = np.array(self.data1)
        d2 = np.array(self.data2)

        #構造係数
        #fu1, con1= self.staticCoef(d1[row:row+self.wins,:], self.wx_m[:,row,col])
        #gu2, con2= self.staticCoef(d2[col:col+self.wins,:], self.wy_m[:,row,col])

        # row colの解釈が異なるため、変更する必要がある, d1, d2の
        fu1, con1= self.staticCoef(d1[col:col+self.wins,:], self.wx_m[:, row, col])
        gu2, con2= self.staticCoef(d2[col+(width-row):col+(width-row)+self.wins,:], self.wy_m[:, row, col])

        print "fu", fu1
        print "gu", gu2
        """
        ds1, ds2, ds3 = self.wx_m.shape
        xl = np.arange(ds1)
        
        self.wx_area.cla()
        self.wx_area.bar(xl, fu1)
        self.wx_area.set_xlim(0, ds1)
        self.wx_area.set_ylim(-1,1)
        self.wx_area.set_title("user1 vec",fontsize=self.fs+1)

        self.wy_area.cla()
        self.wy_area.bar(xl, gu2)
        self.wy_area.set_xlim(0, ds1)
        self.wy_area.set_ylim(-1,1)
        self.wy_area.set_title("user2 vec",fontsize=self.fs+1)

        self.fig.canvas.draw()
        """

    def get_index(self, w):
        js = []
        for i, d in enumerate(w):
            if d != 0:
                js.append(i)
        return js
        
    def calc_sq(self, X):        
        X = X - X.mean(axis=0)
        S = np.cov(X.T, bias=1)
        sq = SLA.sqrtm(SLA.inv(S))
        return sq


    def draw_sig(self, row, col, width):
        print "draw_sig:",row,col
        d1 = np.array(self.data1)
        d2 = np.array(self.data2)

        # 可視化されない。なぜ? shapeが(0, 51)になってる??←再現できない...解決されてる??
        # 重みの入ってるjointのインデックス
        j1, j2 = self.get_index(self.wx_m[:,row,col]), self.get_index(self.wy_m[:,row,col])
        #j1, j2は,手先などのデータをカットした後のインデックス(全51次元)だから、75次元のデータとマッチングしない
        #print "j1",len(j1),j1
        #print "j2",len(j2),j2
        
        dr1 = d1[col:col+self.wins,:]
        dr2 = d2[col+(width-row):col+(width-row)+self.wins,:]
        ds1 = dr1[:,j1]
        ds2 = dr2[:,j2]
        #ds1 = d1[col:col+self.wins,j1]
        #ds2 = d2[col+(width-row):col+(width-row)+self.wins,j2] #rowとcolを使ってデータのオフセットを表現

        """
        wxlist=[0]*len(self.sidx)
        for (i, d) in enumerate(self.sidx):
            self.wxlist[i] = self.wx_m[d, row, col]
        """
        # 基底変換する
        #wx = np.dot(self.calc_sq(ds1), self.wx_m[j1, row, col])
        #wy = np.dot(self.calc_sq(ds2), self.wy_m[j2, row, col])

        # 正準値を可視化してみる
        wx, wy = self.wx_m[j1, row, col], self.wy_m[j2, row, col]
        #f, g = np.dot(ds1, wx), np.dot(ds2, wy)
        f, g = ds1, ds2
        # f, gはn行1列だから以下が成立
        maxf, maxg = f.max(), g.max()
        rng_max = maxf+1 if maxf > maxg else maxg+1
        minf, ming = f.min(), g.min()
        rng_min = minf-1 if minf < ming else ming-1

        #print "f",f.shape

        # 正準相関の値
        #corr = np.corrcoef(f, g)[0,1]

        print "ds1 std:",np.std(ds1, axis=0)
        print "ds2 std:",np.std(ds2, axis=0)

        #これは間違い。差分の絶対値の合計を取るべき
        #print "ds1", ds1.shape, " sum:",np.sum(np.fabs(ds1), axis=0)
        #print "ds2", ds2.shape, " sum:",np.sum(np.fabs(ds2), axis=0)
        #print "corr(ds1, ds2)", corr
        #print ds2.shape
        
        #ds1_m = np.std(ds1, axis=0).mean()
        #ds2_m = np.std(ds2, axis=0).mean()
        #ds1_m = np.sum(np.fabs(ds1), axis=0)
        #ds2_m = np.sum(np.fabs(ds2), axis=0)

        #rng = 1
        self.sig1_area.cla()
        #self.sig1_area.plot(ds1, label="u1", alpha=0.5)
        self.sig1_area.plot(f, label="u1", alpha=0.5)
        self.sig1_area.set_title("user1 cca:"+str(self.r_m[row][col]), fontsize=self.fs+1)
        self.sig1_area.set_ylim(rng_min, rng_max)

        self.sig2_area.cla()
        #self.sig2_area.plot(ds2, label="u2", alpha=0.5)
        self.sig2_area.plot(g, label="u2", alpha=0.5)
        self.sig2_area.set_title("user2 cca:"+str(self.r_m[row][col]), fontsize=self.fs+1)
        self.sig2_area.set_ylim(rng_min, rng_max)

        """
        X, Y = np.fft.fft(f), np.fft.fft(g)
        self.sig2_area.cla()
        self.sig2_area.plot(X, color="r", label="u1", alpha=0.5)
        self.sig2_area.plot(Y, color="g", label="u2", alpha=0.5)
        """

        self.fig.canvas.draw()


        #多重共線性をチェックするためvifを可視化
        tx = np.array(ds1)
        p1 = np.corrcoef(tx.T)
        ty = np.array(ds2)
        p2 = np.corrcoef(ty.T)
        
        print "tx:",tx.shape, ", ty:",ty.shape
        
        # ここで, tx, tyのshapeが(1,)とかになってると計算不能?
        if tx.shape[1] > 1:
            print "data1 vif:"
            for r in range(p1.shape[0] - 1): #p1.shape[0]
                for c in range(r, p1.shape[1] -1):
                    p = p1[r][c+1]
                    vif = 1/(1-p**2)
                    if vif > 10:
                        print r, c, vif

        if ty.shape[1] > 1:
            print "data2 vif:"
            for r in range(p2.shape[0] - 1): 
                for c in range(r, p2.shape[1] -1):
                    p = p2[r][c+1]
                    vif = 1/(1-p**2)
                    if vif > 10:
                        print r, c, vif




    """
    # user1を横軸に,user2のdelayを縦軸に、長方形のrho図を描く予定
    def draw_rho(self):

        dr, dc = self.r_m.shape
        print "dr, dc", dr, dc
        r_row = dc*2-1
        r_col = dr
        Y, X = np.mgrid[slice(0, r_row+1, 1), slice(0, r_col+1, 1)]
        
        rs = np.zeros((r_col, r_row)) #

        for i in range(dr):
            for j in range(dc):
                #print "i,j", i, j
                rs[i][(dc-1-i)+j] = self.r_m[i][j]
        
        print rs.shape
        img = self.rho_area.pcolor(Y, X, rs, vmin=0.0, vmax=1.0, cmap=cm.gray)
        self.rho_area.set_xlim(0,r_row)
        self.rho_area.set_ylim(0,r_col)
        self.fig.canvas.draw()

    """
    def draw_rho(self):
        dr, dc = self.r_m.shape
        Y, X = np.mgrid[slice(0, dr+1, 1),slice(0, dc+1, 1)]

        #Y, X = np.mgrid[slice(-(dr-1)/2, (dr-1)/2+1, 1),slice(0, dc, 1)]        
        img = self.rho_area.pcolor(X, Y, self.r_m, vmin=0.0, vmax=1.0, cmap=cm.gray)
        
        #if self.cbar == None:
        #    self.cbar = self.fig.colorbar(img)
        #    self.cbar.ax.tick_params(labelsize=fs-1) 
        
        self.rho_area.set_xlim(0,dc)
        self.rho_area.set_ylim(0,dr)
        wid = 10 #とりあえず決め打ちで10ずつ目盛表示
        ticks = [i*wid for i in range(dr/wid+1)]
        labels = [(dr-1)/2-i*wid for i in range(dr/wid+1)]
        self.rho_area.set_yticks(ticks=ticks)
        self.rho_area.set_yticklabels(labels=labels)
        self.rho_area.set_xlabel("user 1")
        self.rho_area.set_ylabel("user 2")
        #self.rho_area.tick_params(labelsize=self.fs)
        #self.rho_area.set_title("rho",fontsize=self.fs+1)
        self.fig.canvas.draw()
    

class CCA(QtGui.QWidget):

    def __init__(self):
        super(CCA, self).__init__()
        #UIの初期化
        self.initUI()

        #ROSのパブリッシャなどの初期化
        rospy.init_node('ccaviz', anonymous=True)
        self.mpub = rospy.Publisher('visualization_marker_array', MarkerArray, queue_size=10)
        self.ppub = rospy.Publisher('joint_diff', PointStamped, queue_size=10)
        self.row, self.col = 0, 0
        #rvizのカラー設定(未)
        self.carray = []
        clist = [[1, 0, 0, 1], [0, 1, 0, 1], [1, 1, 0, 1], [1, 0.5, 0, 1]]
        for c in clist:
            color = ColorRGBA()
            color.r = c[0]
            color.g = c[1]
            color.b = c[2]
            color.a = c[3]
            self.carray.append(color) 

    def initUI(self):
        grid = QtGui.QGridLayout()
        form = QtGui.QFormLayout()
        
        #ファイル入力ボックス
        self.txtSepFile = QtGui.QLineEdit()
        btnSepFile = QtGui.QPushButton('...')
        btnSepFile.setMaximumWidth(40)
        btnSepFile.clicked.connect(self.chooseDbFile)
        boxSepFile = QtGui.QHBoxLayout()
        boxSepFile.addWidget(self.txtSepFile)
        boxSepFile.addWidget(btnSepFile)
        form.addRow('input file', boxSepFile)

        #exec
        boxExec = QtGui.QHBoxLayout()
        btnExec = QtGui.QPushButton('visualize')
        btnExec.clicked.connect(self.doExec)
        boxExec.addWidget(btnExec)
        form.addRow('viz', boxExec)

        #pub        
        boxPub = QtGui.QHBoxLayout()
        self.start_line = QtGui.QLineEdit()
        self.start_line.setText('0')
        self.end_line = QtGui.QLineEdit()
        self.end_line.setText('100')
        btnPub = QtGui.QPushButton('publish')
        btnPub.clicked.connect(self.doPub)
        btnAllPub = QtGui.QPushButton('all pub')
        btnAllPub.clicked.connect(self.doAllPub)

        boxPub.addWidget(btnPub)

        boxPub.addWidget(self.start_line)
        boxPub.addWidget(self.end_line)

        boxPub.addWidget(btnAllPub)
        form.addRow('pub', boxPub)

        # 位相ズレごとの正準ベクトルの可視化        
        boxPhase = QtGui.QHBoxLayout()
        self.p_width = QtGui.QLineEdit()
        self.p_width.setText('30')
        boxPhase.addWidget(self.p_width)

        btnPhase = QtGui.QPushButton('plot')
        btnPhase.clicked.connect(self.seqVecPlot)
        boxPhase.addWidget(btnPhase)

        btnErr = QtGui.QPushButton('errer')
        btnErr.clicked.connect(self.errorPlot)
        boxPhase.addWidget(btnErr)

        form.addRow('vector / error', boxPhase)

        #selected pub
        self.radio1 = QtGui.QRadioButton('True')
        form.addRow('match pub time', self.radio1)

        # matplotlib
        boxPlot = QtGui.QHBoxLayout()
        self.main_frame = QtGui.QWidget()
        self.plot = Plot(self.main_frame)
        boxPlot.addWidget(self.plot.canvas)

        #配置
        grid.addLayout(form,1,0)
        grid.addLayout(boxPlot,2,0)

        self.setLayout(grid)
        #self.resize(400,100)
        self.setWindowTitle("cca window")

        self.show()

    def chooseDbFile(self):
        dialog = QtGui.QFileDialog()
        dialog.setFileMode(QtGui.QFileDialog.ExistingFile)
        if dialog.exec_():
            fileNames = dialog.selectedFiles()
            for f in fileNames:
                self.txtSepFile.setText(f)
                return
        return self.txtSepFile.setText('')


    def updateTable(self):
        self.plot.on_draw(self.r_m, self.wx_m, self.wy_m, self.data1, self.data2, self.wins, self.sidx, self.pre_dtd)

    def set_row_col(self, r, c):
        global row
        global col
        row = r
        col = c

    def doExec(self):
        filename = str(self.txtSepFile.text())
        #print "exec!"
        self.wins, self.frms, self.dtd, self.dts, self.sidx, self.pre_dtd = self.load_params(filename)
        self.r_m, self.wx_m, self.wy_m, self.org1, self.org2 = self.load_results(filename, self.dtd)
        self.data1, self.data2, self.pos1, self.pos2 = self.load_datas_poses(filename)
        self.jidx, self.nidx = self.jidx_nidx_input()

        #dmr:data_max_range, frmr:frame_range, dtr:data_range
        self.dtmr = self.dts - self.wins + 1
        self.frmr = self.dts - self.frms + 1
        self.dtr = self.frms - self.wins + 1
        print "dtmr:",self.dtmr,", frmr:",self.frmr,", dtr:", self.dtr
        
        self.updateTable()
        print "end"


    def load_params(self, filename):
        print "load"+filename
        with h5py.File(filename) as f:            
            wins = f["/prop/wins"].value
            frms = f["/prop/frms"].value
            dtd = f["/prop/dtd"].value
            pre_dtd = f["/prop/pre_dtd"].value
            dts = f["/prop/dts"].value
            sidx = f["/prop/sidx"].value
        return wins, frms, dtd, dts, sidx, pre_dtd

    def load_results(self, filename, dtd):
        print "load"+filename
        with h5py.File(filename) as f: 
            r_m = f["/cca/r/0"].value            
            wx_m = [f["/cca/wx/0/"+str(i)].value for i in range(dtd)]
            wy_m = [f["/cca/wy/0/"+str(i)].value for i in range(dtd)]
            org1 = f["/prop/org1"].value
            org2 = f["/prop/org2"].value
        return  np.array(r_m), np.array(wx_m), np.array(wy_m), org1, org2

    def load_datas_poses(self, filename):
        with h5py.File(filename) as f:  
            fposename = f["/prop/fname"].value
            print "open pose file:",fposename
            fp = open(fposename, 'r')
            jsp = json.load(fp)
            #f.close()
            data1 = f["/cca/data/data1"].value
            data2 = f["/cca/data/data2"].value
            if jsp[0]["datas"][0].has_key("jdata"):
                ps = len(jsp[0]["datas"][0]["jdata"]) 
                poses=[[[[u["datas"][j]["jdata"][p][x]for x in range(3)]for p in range(ps)]for j in range(self.dts)]for u in jsp]
            else:
                print "no poses"
                poses = [[0],[0]]
        return data1, data2, poses[0], poses[1]

    def jidx_nidx_input(self):
        jidx = [[3, 2, 20, 1, 0],
                [20, 4, 5, 6],
                [20, 8, 9, 10],
                [0, 12, 13, 14],
                [0, 16, 17, 18]
            ]
        nidx = []
        for sid in self.sidx:
            if sid % 3 == 0:
                nidx.append(sid/3)
        return jidx, nidx

    def doPub(self):        
        r, c = row, col
        cor = self.r_m[r, c]

        print "---play back start---"
        poses, weights, orgs = [], [], []

        #ここでweightを構造係数に変換するか?
        weights.append(self.wx_m[:,r,c])
        weights.append(self.wy_m[:,r,c])

        orgs.append(self.org1)
        orgs.append(self.org2)

        offset = -(r-(self.frms+1))
        # row, colの解釈が異なるので、変更する
        if self.radio1.isChecked(): 
            print "time match"
            poses.append(self.pos1[c:c+self.wins])
            poses.append(self.pos2[c+offset:c+offset+self.wins])
            print "c, c+offset",c, c+offset
            self.pubViz(0, 0, cor, weights, poses, self.wins, orgs)
        else:
            print "time miss match"
            if offset>0: 
                poses.append(self.pos1[c:c+offset+self.wins])
                poses.append(self.pos2[c:c+offset+self.wins])
                self.pubViz(c, c+offset, cor, weights, poses, self.wins, orgs)
            else: 
                poses.append(self.pos1[c+offset:c+self.wins])
                poses.append(self.pos2[c+offset:c+self.wins])
                self.pubViz(c, c+offset, cor, weights, poses, self.wins, orgs)
        print "---play back end---"
 
    def doAllPub(self):
        start = int(self.start_line.text()) if int(self.start_line.text()) >= 0 else 0
        end = int(self.end_line.text()) if int(self.end_line.text()) <= self.dts else self.dts

        print "pub area",start,"-",end
        poses, weights, orgs = [], [], []
        
        orgs.append(self.org1)
        orgs.append(self.org2)
        poses.append(self.pos1[start:end])
        poses.append(self.pos2[start:end])
        self.pubViz(0, 0, 0, weights, poses, self.wins, orgs)


    def rviz_obj(self, obj_id, obj_ns, obj_type, obj_size, obj_color=0, obj_life=0):
        obj = Marker()
        obj.header.frame_id, obj.header.stamp = "camera_link", rospy.Time.now()
        obj.ns, obj.action, obj.type = str(obj_ns), 0, obj_type
        obj.scale.x, obj.scale.y, obj.scale.z = obj_size[0], obj_size[1], obj_size[2]
        obj.color = self.carray[obj_color]
        obj.lifetime = rospy.Duration.from_sec(obj_life)
        obj.pose.orientation.w = 1.0
        return obj

    def set_point(self, pos, addx=0, addy=0, addz=0):
        pt = Point()            
        pt.x, pt.y, pt.z = pos[0]+addx, pos[1]+addy, pos[2]+addz
        return pt

    def seqVecPlot(self):
        r, c = row, col
        phase = row
        start = col
        end = start + int(self.p_width.text())

        fig = plt.figure()
        print self.wx_m.shape
        # xv:関節, pv:位相, yv:データ長
        xv, pv, yv = self.wx_m.shape
        xs = np.arange(xv)
        ax1 = fig.add_subplot(111, projection='3d')   
        # 例外処理:endがwx_mの範囲を超えたときの
        for num in range(start, end):
            ys=np.sqrt(self.wx_m[:, phase, num]**2)
            ax1.bar(xs, ys, zs=num, zdir='y', color="r", alpha=0.8)
            if self.dtmr < num:
                print "data max range:", self.dtmr
                break
        ax1.set_xlim(0,51)
        ax1.set_ylim(start, end)
        plt.show()

    def errorPlot(self):
        width = int(self.p_width.text())
        print "error, width:",width
        err_size = self.dtmr - width
        buff = np.zeros((self.frms*2+1, err_size))

        for phase in range(self.frms*2+1):
            for start in range(err_size):
                end = start + width
                buff_tmp = np.zeros(self.dtd)
                for num in range(start, end):
                     buff_tmp += np.fabs(self.wx_m[:,phase,num+1] - self.wx_m[:,phase,num])
                buff[phase, start] = buff_tmp.mean()

        dr, dc = buff.shape
        Y, X = np.mgrid[slice(0, dr+1, 1),slice(0, dc+1, 1)]
        plt.pcolor(X, Y, buff, vmin=0.0, vmax=10)
        plt.xlim(0,dc)
        plt.ylim(0,dr)
        plt.colorbar()
        plt.show()
        #print buff
        


    def pubViz(self, r, c, cor, wts, poses, wins, orgs):
        print "r, c", r, c
        rate = rospy.Rate(17)
        offset = c if (r>c) else r

        for i in range(len(poses[0])):
            sq = i+offset
            print "frame:",sq
            msgs = MarkerArray()
            tmsg = self.rviz_obj(10, 'f10', 9, [0.1, 0.1, 0.1], 0)
            tmsg.pose.position.x,tmsg.pose.position.y,tmsg.pose.position.z=0,0,0
            tmsg.text = "c:"+str(round(cor, 3))+", f:"+str(sq)
            msgs.markers.append(tmsg) 

            for u, pos in enumerate(poses):
                # points
                pmsg = self.rviz_obj(u, 'p'+str(u), 7, [0.03, 0.03, 0.03], 0)
                pmsg.points = [self.set_point(p) for p in np.array(pos[i])[self.nidx]]
                msgs.markers.append(pmsg)
                
                """
                # origin points
                omsg = self.rviz_obj(u, 'o'+str(u), 7, [0.03, 0.03, 0.03], 1)
                omsg.points = [self.set_point(orgs[u])]
                msgs.markers.append(omsg)
                """

                # lines
                lmsg = self.rviz_obj(u, 'l'+str(u), 5, [0.005, 0.005, 0.005], 2)
                for jid in self.jidx:
                    for pi in range(len(jid)-1):
                        for add in range(2):
                            lmsg.points.append(self.set_point(pos[i][jid[pi+add]])) 
                msgs.markers.append(lmsg)
                
                # text
                tjs = 0.1
                tmsg = self.rviz_obj(u, 't'+str(u), 9, [tjs, tjs, tjs], 0)
                tmsg.pose.position = self.set_point(pos[i][3], addy=tjs, addz=tjs)
                tmsg.text = "user_"+str(u+1)
                msgs.markers.append(tmsg) 
                
                # arrow
                if len(wts) != 0:
                    amsg = self.rviz_obj(u, 'a'+str(u), 5,  [0.005, 0.005, 0.005], 1) 
                    num = 0.3
                    df = num/max(np.fabs(wts[u]))
                    for (ni, nid) in enumerate(self.nidx):
                        amsg.points.append(self.set_point(pos[i][nid]))
                        amsg.points.append(self.set_point(pos[i][nid], addx=wts[u][ni*3]*df))
                        amsg.points.append(self.set_point(pos[i][nid]))
                        amsg.points.append(self.set_point(pos[i][nid], addy=wts[u][ni*3+1]*df))
                        amsg.points.append(self.set_point(pos[i][nid]))
                        amsg.points.append(self.set_point(pos[i][nid], addz=wts[u][ni*3+2]*df))
                    msgs.markers.append(amsg) 
              
                if u == 0 and sq > r and sq < r+wins:    
                    #print "now interaction", u
                    npmsg = self.rviz_obj(u, 'np'+str(u), 7, [0.05, 0.05, 0.05], 3, 0.1)
                    npmsg.points = [self.set_point(p) for p in np.array(pos[i])[self.nidx]]
                    msgs.markers.append(npmsg)
                if u == 1 and sq > c and sq < c+wins:   
                    #print "now interaction", u 
                    npmsg = self.rviz_obj(u, 'np'+str(u), 7, [0.05, 0.05, 0.05], 3, 0.1)  
                    npmsg.points = [self.set_point(p) for p in np.array(pos[i])[self.nidx]]
                    msgs.markers.append(npmsg)
                
            self.mpub.publish(msgs)
            rate.sleep()

def main():
    app = QtGui.QApplication(sys.argv)
    corr = CCA()
    sys.exit(app.exec_())

if __name__=='__main__':
    main()
