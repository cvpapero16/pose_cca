#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import sys
import math
import json
import numpy as np
import h5py

from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import *
from PyQt4.QtGui  import *

import rospy
from visualization_msgs.msg import MarkerArray
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from geometry_msgs.msg import PointStamped
from std_msgs.msg import ColorRGBA


parser = argparse.ArgumentParser()
parser.add_argument('--filename', '-f', default='',
                    help='Input joints file')
args = parser.parse_args()

class Correlation(QtGui.QWidget):

    def __init__(self):
        #self.jsonInput()

        super(Correlation, self).__init__()
        self.initUI()
        rospy.init_node('correlation', anonymous=True)
        self.mpub = rospy.Publisher('visualization_marker_array', MarkerArray, queue_size=10)
        self.ppub = rospy.Publisher('joint_diff', PointStamped, queue_size=10)

        self.carray = []
        clist = [[1,1,0,1],[0,1,0,1],[1,0,0,1]]
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

        self.winSizeBox = QtGui.QLineEdit()
        self.winSizeBox.setText('30')
        self.winSizeBox.setAlignment(QtCore.Qt.AlignRight)
        self.winSizeBox.setFixedWidth(100)
        form.addRow('window size', self.winSizeBox)

        self.AFrameBox = QtGui.QLineEdit()
        self.AFrameBox.setText('0')
        self.AFrameBox.setAlignment(QtCore.Qt.AlignRight)
        self.AFrameBox.setFixedWidth(100)
        form.addRow('A User Frame', self.AFrameBox)

        self.BFrameBox = QtGui.QLineEdit()
        self.BFrameBox.setText('0')
        self.BFrameBox.setAlignment(QtCore.Qt.AlignRight)
        self.BFrameBox.setFixedWidth(100)
        form.addRow('B User Frame', self.BFrameBox)

        boxCtrl = QtGui.QHBoxLayout()
        btnExec = QtGui.QPushButton('exec')
        btnExec.clicked.connect(self.doExec)
        boxCtrl.addWidget(btnExec)

        grid.addLayout(form,1,0)
        grid.addLayout(boxCtrl,2,0)

        self.setLayout(grid)
        self.resize(400,100)
        
        self.setWindowTitle("joint select window")
        self.show()


    def json_pose_input(self, filename):
        f = open(filename, 'r')
        jD = json.load(f)
        f.close()
        #pose
        ds, dd, dp = len(jD[0]["datas"]), len(jD[0]["datas"][0]["jdata"]), len(jD[0]["datas"][0]["jdata"][0])  
        datas=[[[u["datas"][t]["jdata"][p][i]for i in range(dp) for p in range(dd)] for t in range(ds)] for u in jD]  

        return np.array(datas[0]), np.array(datas[1]), ds, dd*dp



    def jsonInput(self, filename):
        f = open(filename, 'r');
        jD = json.load(f)
        f.close()

        datas = []
        ds, dd, dp = len(jD[0]["datas"]), len(jD[0]["datas"][0]["jdata"]), len(jD[0]["datas"][0]["jdata"][0])

        for user in jD:

            pobj = []
            for s in range(ds):
                pl = []
                for d in range(dd):
                    for p in range(dp):
                        pl.append(user["datas"][s]["jdata"][d][p])
                pobj.append(pl)
            datas.append(pobj)

        print np.array(datas[1]).shape

        return np.array(datas[0]), np.array(datas[1]), ds, dd*dp

    def json_pose_input(self, filename):
        f = open(filename, 'r')
        jD = json.load(f)
        f.close()
        #pose
        ds, dd, dp = len(jD[0]["datas"]), len(jD[0]["datas"][0]["jdata"]), len(jD[0]["datas"][0]["jdata"][0])  
        datas=[[[u["datas"][t]["jdata"][p][i]for i in range(dp) for p in range(dd)] for t in range(ds)] for u in jD]  

        return np.array(datas[0]), np.array(datas[1]), ds, dd*dp

    def hdf_input(self, filename):
        print "load", filename
        with h5py.File(filename) as f:
            data1 = f["/cca/data/data1"].value
            data2 = f["/cca/data/data2"].value
        return data1, data2


    def doExec(self):
        print "exec!"

        #self.data1, self.data2, self.dts, self.dtd = self.jsonInput(args.filename)
        
        self.data1, self.data2 = self.hdf_input(args.filename)

        self.winSize = int(self.winSizeBox.text())
        #self.maxRange = self.dataSize - self.winSize

        Ast = float(self.AFrameBox.text())
        Bst = float(self.BFrameBox.text())

        self.pubViz(self.data1, self.data2, Ast, Bst)

        print "end"

    def rviz_obj(self, obj_id, obj_ns, obj_type, obj_size, obj_color=0, obj_life=0):
        obj = Marker()
        obj.header.frame_id, obj.header.stamp = "camera_link", rospy.Time.now()
        obj.ns, obj.action, obj.type = str(obj_ns), 0, obj_type
        obj.scale.x, obj.scale.y, obj.scale.z = obj_size, obj_size, obj_size
        obj.color = self.carray[obj_color]
        obj.lifetime = rospy.Duration.from_sec(obj_life)
        obj.pose.orientation.w = 1.0
        return obj

    def set_point(self, pos, addx=0, addy=0, addz=0):
        pt = Point()  
        print "pos", pos
        pt.x, pt.y, pt.z = pos[0]+addx, pos[1]+addy, pos[2]+addz
        return pt

    def pubViz(self, data1, data2, ast, bst):

        rate = rospy.Rate(10)
        js = 0.03
        
        for i in range(self.winSize):

            msgs = MarkerArray()
            
            for u,(data, st) in enumerate(zip([data1, data2], [ast, bst])):
                pmsg = self.rviz_obj(u, 'p'+str(u), 7, 0.03, u)

                #ジョイントポイントを入れる処理
                #print np.array(data[st+i]).shape
                for j in range(len(data[st+i])):
                    if j%3 == 0:
                        point = Point()
                        point.x = data[st+i][j]
                        point.y = data[st+i][j+1]
                        point.z = data[st+i][j+2]
                        pmsg.points.append(point) 
                #pmsg.points = [self.set_point(p) for p in data[st+i]]
                msgs.markers.append(pmsg)    
            
            """
            msg = Marker()
            msg.header.frame_id = 'camera_link'
            msg.header.stamp = rospy.Time.now()
            msg.ns = 'j2'
            msg.action = 0
            msg.id = 2
            msg.type = 8
            msg.scale.x = js
            msg.scale.y = js
            msg.scale.z = js
            msg.color = self.carray[1]

            for j2 in range(len(self.pdata[0][bst+i])):
                point = Point()
                point.x = self.pdata[1][bst+i][j2][0]
                point.y = self.pdata[1][bst+i][j2][1]
                point.z = self.pdata[1][bst+i][j2][2]
                msg.points.append(point) 
            msg.pose.orientation.w = 1.0
            
            msgs.markers.append(msg)
            """
            self.mpub.publish(msgs)
            rate.sleep()

def main():
    app = QtGui.QApplication(sys.argv)
    corr = Correlation()
    sys.exit(app.exec_())


if __name__=='__main__':
    main()
