from flask import Flask, request, redirect, url_for, render_template
import cv2
import math
import time
import pymysql
# import numpy as np
import pyopenpose as op
import base64


def video_detect(path, db):
    # def Length(A, B):
    #     return np.sqrt((Pose[n][A][0] - Pose[n][B][0]) ** 2 + (Pose[n][A][1] - Pose[n][B][1]) ** 2)

    # def Angle(A, B, C):  # Body 25的三个坐标求角度 求∠B
    #     a = Length(A, B)
    #     b = Length(B, C)
    #     c = Length(A, C)
    #     return (np.arccos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b))) * 180 / np.pi

    def fillin_db(db, MaxAngle, HoldTime, CompAngle):
        week = str(time.strftime("%A", time.localtime()))
        cursor = db.cursor()
        sql = "INSERT INTO " + week + "(MaxAngle, HoldTime, CompAngle) VALUES (%s, %s, %s);"
        cursor.execute(sql, (MaxAngle, HoldTime, CompAngle))
        db.commit()


    cap = cv2.VideoCapture(path)
    '''配置模型'''
    params = dict()  # 创建一个字典
    params["model_folder"] = "models/"  # 修改路径
    params["model_pose"] = "BODY_25"  # 选择pose模型
    params["number_people_max"] = 2  # 最大检测人体数
    params["display"] = 2  # 2D模式

    opWrapper = op.WrapperPython()
    opWrapper.configure(params)  # 导入上述参数
    opWrapper.start()
    '''配置模型'''

    '''初始化'''
    '''SSF'''
    n = 0  # 默认user=0
    j = 0; j1 = 0; j2 = 0; j3 = 0  # 4种状态表明arms up or down
    SSF = 0  # 计数
    SSF_Angle = 270  # 实际角度
    SSF_Max = 0  # 抬手最大值
    SSF_Time = 0  # 测量间隔
    SSF_Num = 5  # 间隔帧数，30帧为1s
    SSF_Hold = 0  # hold时间
    SSF_Flag = 0  # 计时阀门关闭
    SSF_start = 0  # 开始时间
    SSF_Sum = 0  # 角度差值和
    helper = 'None'
    string2 = 'None'
    Max_comps = 0


    while True:
        ret, frame = cap.read()
        if not ret:
            break
        imageToProcess = cv2.resize(frame, (800, 600), )  # 图像尺寸变换
        imageToProcess = cv2.cvtColor(imageToProcess, cv2.COLOR_RGB2BGR)  # RGB转换为BGR

        '''计算关节点'''
        datum = op.Datum()
        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        Pose = datum.poseKeypoints  # 关键点存在的地方,若无人像，输出None
        BGR = datum.cvOutputData  # imageToProcess 已识别后的图像
        '''计算关节点'''

        '''模型测量'''
        if str(Pose) != 'None':

            '''Helper'''
            if Pose.shape[0] == 1:
                n = 0  # Only one person
                helper = 'Yes!'
            else:
                if Pose[1][8][2] != 0 and Pose[1][9][2] != 0 and Pose[1][12][2] != 0:
                    if Pose[0][8][2] != 0 and Pose[0][9][2] != 0 and Pose[0][12][2] != 0:
                        if Pose[0][8][1] > Pose[1][8][1] and Pose[0][9][1] > Pose[1][9][1] and Pose[0][12][1] > \
                                Pose[1][12][1]:
                            n = 0  # 0先生是坐着的
                        else:
                            n = 1
                    else:
                        n = 1
                else:
                    n = 0
                helper = 'No'
            '''Helper'''  # n = user

            '''Compensation'''
            if j == 1 and j1 == 1 and j2 == 1 and j3 == 1:
                if abs(Max_comps) < abs(math.atan((Pose[n][1][0] - Pose[n][8][0]) / (Pose[n][8][1] - Pose[n][1][1])) * 180 / math.pi):
                    Max_comps = round(math.atan((Pose[n][1][0] - Pose[n][8][0]) / (Pose[n][8][1] - Pose[n][1][1])) * 180 / math.pi, 0)
                if abs(Max_comps) < 10:
                    string2 = 'Straight!'
                    Max_comps = 0
                else:
                    if Pose[n][1][0] > Pose[n][8][0]:
                        string2 = 'Front Compensation! ' + str(Max_comps)
                    else:
                        string2 = 'Back Compensation! ' + str(Max_comps)

            '''Compensation'''

            '''SSF'''
            # 右手臂存在
            if Pose[n][4][2] != 0 and Pose[n][3][2] != 0 and Pose[n][2][2] != 0 and Pose[n][1][2] != 0 and Pose[n][8][2] != 0:
                '''Num'''
                SSF_Deviation = abs(SSF_Angle - math.atan(abs((Pose[n][2][1]-Pose[n][3][1]) / (Pose[n][3][0]-Pose[n][2][0]))) * 180 / math.pi)
                SSF_Angle = math.atan(abs((Pose[n][2][1]-Pose[n][3][1]) / (Pose[n][3][0]-Pose[n][2][0]))) * 180 / math.pi
                if j == 1 and j1 == 1 and j2 == 1 and j3 == 1:
                    SSF_Max = round(max(SSF_Max, SSF_Angle + Max_comps), 0)  # 抬手时找角度最小
                if Pose[n][4][1] < Pose[n][2][1]:  # hands up
                    j3 = 1
                else:  # hands down
                    j3 = 0
                if j == 0 and j1 == 0 and j2 == 1 and j3 == 1:
                    SSF = SSF + 1
                j = j1
                j1 = j2
                j2 = j3
                '''Num'''

                '''Hold time'''  # 一帧0.033s / 一秒30帧
                if j == 1 and j1 == 1 and j2 == 1 and j3 == 1:  # 1111 hands up
                    SSF_Time = SSF_Time + 1  # 帧数
                    if SSF_Time % SSF_Num == 0:  # 开始判定
                        if SSF_Sum / SSF_Num < 1.5:  # if hold？ hold < 6度
                            if SSF_Flag == 0:
                                # SSF_start = time.time()
                                SSF_start = SSF_Time  # 开始的帧数
                                SSF_Flag = 1
                        else:
                            SSF_Flag = 0

                        SSF_Sum = 0

                    else:  # 角度均值
                        SSF_Sum = SSF_Deviation + SSF_Sum

                # print(SSF_Flag)
                if SSF_Flag == 1 and SSF_Time - SSF_start > 20:  # 找到最后的hold值
                    SSF_Hold = round((SSF_Time - SSF_start) * 0.033, 2)
                '''Hold time'''

                '''fill in database'''
                if j == 1 and j1 == 1 and j2 == 0 and j3 == 0:  # 1100放手
                    fillin_db(db, str(SSF_Max + 90), str(SSF_Hold), str(Max_comps))
                    SSF_Hold = 0
                    SSF_Max = 0
                    Max_comps = 0
                '''fill in database'''

            else:
                BGR = cv2.putText(BGR, 'Move left', (5, 910), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)
            '''SSF'''
            ###################################################################################################

        else:
            BGR = cv2.putText(BGR, 'Move into camera', (5, 940), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

        '''Video Display'''
        string0 = 'SSF: ' + str(SSF) + ' Hold:' + str(SSF_Hold) + 's'
        BGR = cv2.putText(BGR, string0, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)
        BGR = cv2.putText(BGR, 'Angle: ' + str(90+SSF_Max), (5, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)
        BGR = cv2.putText(BGR, string2, (5, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)
        BGR = cv2.putText(BGR, helper, (5, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)

        imageRGB = cv2.cvtColor(BGR, cv2.COLOR_BGR2RGB)
        cv2.imshow('Openpose', imageRGB)
        cv2.waitKey(1)
        '''Video Display'''

    cap.release()
    cv2.destroyAllWindows()
    # 次数， hold, max angle, compensation, helper
    return


def Connect_db():
    db = pymysql.connect(host='database-1.cwxo5gmzuasx.ap-southeast-1.rds.amazonaws.com', port=3306,
                         user='admin', password="chenliyu5078", db='rehab')  # 连接数据库
    return db




app = Flask(__name__)


@app.route('/')
def upload_form():
    return render_template('index.html')


@app.route('/', methods=["POST"])
def results():
    file = request.form.get('userfile')  # 现在file是视频转的字符串，解码就可
    file = file.split(",")[1]  # 去掉逗号前不属于编码的东西
    file = base64.b64decode(file)  # altchars=None, validate=False) # 解码后是字节流
    with open('video.mp4', 'wb+') as video:
        video.write(file)
    db = Connect_db()  # 连接数据库
    video_detect('video.mp4', db)
    db.close()  # 断开连接
    return redirect(url_for('show'), code=302)


@app.route('/show') # 读取数据库信息，加载折线图
def show():
    db = Connect_db()
    cursor = db.cursor()
    # week = str(time.strftime("%A", time.localtime()))
    week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    MaxAngle_data, HoldTime_data, CompAngle_data = [], [], []

    # SQL 查询语句
    for w in week:
        sql = "SELECT * FROM " + w
        cursor.execute(sql)
        # 获取该星期所有记录列表
        results = cursor.fetchall()
        times, MaxAngle, HoldTime, CompAngle = [], [], [], []
        for row in results:
            times.append(row[0])
            MaxAngle.append(row[1])
            HoldTime.append(row[2])
            CompAngle.append(row[3])
        MaxAngle_data.append(MaxAngle)
        HoldTime_data.append(HoldTime)
        CompAngle_data.append(CompAngle)

    return render_template('histogram.html', MA=MaxAngle_data, HT=HoldTime_data, CA=CompAngle_data)


if __name__ == "__main__":
    app.run(debug=True, host='192.168.1.115', port=5000, ssl_context="adhoc")

