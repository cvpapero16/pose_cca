#!/usr/bin/python
# -*- coding: utf-8 -*-

"""

完成版

関節を標準化するために回転させる
作成中、、法線方向に回転させて、最後に、肩の軸をy軸と一致させる

"""

import numpy as np
import json

import rospy
from visualization_msgs.msg import MarkerArray
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point

poses = [[[1.9050928354263306,0.44349154829978943,-0.088180467486381531],[1.9142193794250488,0.46227467060089111,0.23649357259273529],[1.9097474813461304,0.47805970907211304,0.54749792814254761],[1.930761456489563,0.42726427316665649,0.67671698331832886],[2.0365803241729736,0.40347534418106079,0.45558053255081177],[2.0997688770294189,0.3573468029499054,0.19231367111206055],[2.115877628326416,0.34065654873847961,-0.080284565687179565],[2.0768027305603027,0.34799712896347046,-0.20576328039169312],[1.7598657608032227,0.5541113018989563,0.43695375323295593],[1.6558398008346558,0.59034109115600586,0.1751246452331543],[1.6155223846435547,0.52942967414855957,-0.074711523950099945],[1.636675238609314,0.50062382221221924,-0.13998782634735107],[1.9211001396179199,0.40065324306488037,-0.094267815351486206],[1.9453847408294678,0.46581679582595825,-0.4900488555431366],[1.9076783657073975,0.52413022518157959,-0.86437302827835083],[1.8322368860244751,0.44181981682777405,-0.88594180345535278],[1.814568042755127,0.46921765804290771,-0.078748084604740143],[1.7523746490478516,0.54198282957077026,-0.47947883605957031],[1.7232530117034912,0.58853596448898315,-0.85602068901062012],[1.6730194091796875,0.55633044242858887,-0.91331547498703003],[1.9130818843841553,0.4746394157409668,0.47159579396247864],[2.0560784339904785,0.34874823689460754,-0.24304404854774475],[2.0202500820159912,0.35866641998291016,-0.1974298357963562],[1.6382136344909668,0.49222531914710999,-0.22326073050498962],[1.6652796268463135,0.45079827308654785,-0.1219027191400528]]]


class JointsStd():
    def __init__(self):
        #ROSのパブリッシャなどの初期化
        #rospy.init_node('ccaviz', anonymous=True)

        fname = "/home/uema/catkin_ws/src/pose_cca/datas/20160520_a2.json"
        poses1, poses2, dts, dtd = self.json_pose_input(fname)
        self.poses1, self.poses2 = self.select_datas(poses1, poses2)
        #print poses1.shape

        self.mpub = rospy.Publisher('visualization_marker_array', MarkerArray, queue_size=10)
        self.carray = []
        clist = [[1, 0, 0, 1], [0, 1, 0, 1], [1, 1, 0, 1], [1, 0.5, 0, 1]]
        for c in clist:
            color = ColorRGBA()
            color.r = c[0]
            color.g = c[1]
            color.b = c[2]
            color.a = c[3]
            self.carray.append(color)
        #self.sidx = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,24,25,26,27,28,29,30,31,32,36,37,38,39,40,41,42,43,44,48,49,50,51,52,53,54,55,56,60,61,62]
        #self.jidx, self.nidx = self.jidx_nidx_input()

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


    def normalize_data(self, data, os):

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
            print org.shape
            shift = s - org
            shift_pose = []
            for dt in data:
                dt += shift
                shift_pose.append(dt)
            return shift_pose

        def data_set(data, os):

            """
            ds = []
            for d in range(len(data)/os):
                ds.append([data[d*os], data[d*os+1], data[d*os+2]])
            """
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
        rot_pose_norm = calc_rot_pose(rot_pose, th_z, th_y, np.array([0,0,0]))
        # 変換後のorg
        rot_org = calc_org(rot_pose_norm, s_l_id, s_r_id, spn_id, os)
        # orgのxを特定の値に移動する
        s = [0,0,0]
        shift_pose = calc_shift_pose(rot_pose_norm, rot_org, s)
        shift_org = calc_org(shift_pose, s_l_id, s_r_id, spn_id, os)

        print data_reset(shift_pose).shape
        return shift_pose, shift_org 


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

    def select_datas(self, data1, data2):
        self.sIdx = [0,1,2,3,4,5,6,8,9,10,12,13,14,16,17,18,20]
        new_sid = [sid*3+i  for sid in self.sIdx for i in xrange(3)]
        self.sIdx = new_sid
        #print self.sIdx
        return data1[:,self.sIdx],  data2[:,self.sIdx]

    def pubViz(self):

        rate = rospy.Rate(10)

        msgs = MarkerArray()

        # x, y, zの軸を描画
        lmsg = self.rviz_obj(0, 'l'+str(0), 5, [0.005, 0.005, 0.005], 2)       
        lmsg.points.append(self.set_point([0,0,0])) 
        lmsg.points.append(self.set_point([0,0,0], addx=1))
        lmsg.points.append(self.set_point([0,0,0])) 
        lmsg.points.append(self.set_point([0,0,0], addy=1))
        lmsg.points.append(self.set_point([0,0,0])) 
        lmsg.points.append(self.set_point([0,0,0], addz=1))
        msgs.markers.append(lmsg)


        # この時点でposes1, poses2は(4720, 51)の行列になっている

        os = 3
        #for u, pos in enumerate(poses1):
        
        #print pos.shape
        u = 0
        points = []
        pos = self.poses2[0]
        for p in range(len(pos)/os):
            pmsg = self.rviz_obj(u, 'p'+str(u), 7, [0.03, 0.03, 0.03], 0)
            points.append(self.set_point([pos[p*os], pos[p*os+1], pos[p*os+2]]))
            pmsg.points = points
        msgs.markers.append(pmsg)
        #print pmsg.points
        
        #org, normal, rot_pose = self.normalize_datas(pos)
        rot_pose, org  = self.normalize_data(pos, 3)
        
        # origin points
        omsg = self.rviz_obj(u, 'o'+str(u), 7, [0.03, 0.03, 0.03], 1)
        omsg.points = [self.set_point(org)]
        msgs.markers.append(omsg)
        
        
        # normal line
        #nlmsg = self.rviz_obj(u, 'nl'+str(u), 5, [0.005, 0.005, 0.005], 2)       
        # nlmsg.points.append(self.set_point(org)) 
        #nlmsg.points.append(self.set_point([0,0,0])) 
        #nlmsg.points.append(self.set_point(normal))
        #msgs.markers.append(nlmsg)
        
        
        # rot_pose
        rpmsg = self.rviz_obj(u, 'rp'+str(u), 7, [0.03, 0.03, 0.03], 3)
        #pmsg.points = [self.set_point(p) for p in np.array(pos[0])[self.nidx]]
        rpmsg.points = [self.set_point(p) for p in np.array(rot_pose)]
        msgs.markers.append(rpmsg)
        
        
        self.mpub.publish(msgs)
        rate.sleep()


# Main function.
if __name__ == '__main__':
    # Initialize the node and name it.
    rospy.init_node('joints_std')
    # Go to class functions that do all the heavy lifting. Do error checking.
    try:
        js = JointsStd()
        while not rospy.is_shutdown():
            js.pubViz()
    except rospy.ROSInterruptException: pass
