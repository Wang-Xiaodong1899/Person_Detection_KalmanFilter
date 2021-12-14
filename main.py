import cv2
import numpy as np
#状态向量
stateSize = 6
#观测向量
measSize = 4
coutrSize = 0
kf = cv2.KalmanFilter(stateSize,measSize,coutrSize)
state = np.zeros(stateSize, np.float32)#[x,y,v_x,v_y,w,h],左上角x, 左上角y，速度，高宽
meas = np.zeros(measSize, np.float32)#[z_x,z_y,z_w,z_h]
procNoise = np.zeros(stateSize, np.float32)

#状态转移矩阵
cv2.setIdentity(kf.transitionMatrix)#生成单位矩阵
# [1 0 dT 0  0 0]
# [0 1 0  dT 0 0]
# [0 0 1  0  0 0]
# [0 0 0  1  0 0]
# [0 0 0  0  1 0]
# [0 0 0  0  0 1]
#观测矩阵
# [1 0 0 0 0 0]
# [0 1 0 0 0 0]
# [0 0 0 0 1 0]
# [0 0 0 0 0 1]
kf.measurementMatrix = np.zeros((measSize,stateSize),np.float32)
kf.measurementMatrix[0,0]=1.0
kf.measurementMatrix[1,1]=1.0
kf.measurementMatrix[2,4]=1.0
kf.measurementMatrix[3,5]=1.0

#预测噪声
# [Ex 0 0 0 0 0]
# [0 Ey 0 0 0 0]
# [0 0 Ev_x 0 0 0]
# [0 0 0 Ev_y 0 0]
# [0 0 0 0 Ew 0]
# [0 0 0 0 0 Eh]
cv2.setIdentity(kf.processNoiseCov)
kf.processNoiseCov[0,0] = 1e-2
kf.processNoiseCov[1,1] = 1e-2
kf.processNoiseCov[2,2] = 5.0
kf.processNoiseCov[3,3] = 5.0
kf.processNoiseCov[4,4] = 1e-2
kf.processNoiseCov[5,5] = 1e-2

#测量噪声
cv2.setIdentity(kf.measurementNoiseCov)
# for i in range(len(kf.measurementNoiseCov)):
#     kf.measurementNoiseCov[i,i] = 1e-1

video_cap = cv2.VideoCapture('IMG.MOV')
# 视频输出
fps = video_cap.get(cv2.CAP_PROP_FPS) #获得视频帧率，即每秒多少帧
size = (int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
videoWriter = cv2.VideoWriter('res_kalman.mp4' ,cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, size)
ticks = 0 
i=0
cnt = 0
found = False
notFoundCount = 0
prePointCen = []
meaPointCen = []
while(True):
    ret, frame = video_cap.read()
    if ret is False:
        break
    # cv2.imshow('frame',frame)
    # cv2.waitKey(1)
    precTick = ticks  
    ticks = float(cv2.getTickCount())  
    res = frame.copy()
    # dT = float(1/fps)  
    dT = float((ticks - precTick)/cv2.getTickFrequency()) 
    shape = frame.shape
    # print(shape)
    pos = (int(shape[1])-300, 80)
    cnt +=1
    # print(cnt)
    cv2.putText(res, "Reserve: Wang-Xiaodong", pos, cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255), 2)
    pos = (int(shape[1])-300, 60)
    cv2.putText(res, "wangxd220@gmail.com", pos, cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255), 2)
    if(found):
        kf.transitionMatrix[0,2] = dT
        kf.transitionMatrix[1,3] = dT
        state = kf.predict()    #kalman校正后的数据
        width = state[4]
        height = state[5]
        x_left = state[0]#左上角横坐标
        y_left = state[1]  #左上角纵坐标
        x_right = state[0] + width
        y_right = state[1] + height
        # print("Measure matrix: ", meas)
        # print("Predict matrix: ","[",x_left[0],y_left[0],width[0],height[0],"]")
        cv2.rectangle(res,(int(x_left[0]),int(y_left[0])),(int(x_right[0]),int(y_right[0])),(255,255,255),2)#白色框预测值
        name = "P: (" + str(int(x_left)) + "," + str(int(y_left)) + ")"
        cv2.putText(res, name, (int(x_left) -5, int(y_left) - 5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255), 2)

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    rangeRes = cv2.inRange(gray, 0, 100)
    kernel = np.ones((3, 3), np.uint8)
    # 腐蚀膨胀
    rangeRes = cv2.erode(rangeRes, kernel, iterations=2)
    rangeRes = cv2.dilate(rangeRes, kernel, iterations=2)
    cv2.waitKey(1)
    contours = cv2.findContours(rangeRes.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[-2]
    person = []
    personBox = []
    c = max(contours, key = cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)  #轮廓点
    cv2.rectangle(res,(x,y),(x+w,y+h),(0,0,255),2)##红色框观测值
    name = "M: (" + str(x) + "," + str(y) + ")"
    cv2.putText(res, name, (int(x) -5, int(y) - 5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255), 2)
    person.append(c)
    personBox.append([x, y, w, h])
    # print( "person found:", len(personBox))
    if(len(person) == 0):
        notFoundCount += 1
        print("notFoundCount",notFoundCount)

    else:
        notFoundCount = 0
        meas[0] = personBox[0][0]
        meas[1] = personBox[0][1]
        meas[2] = float(personBox[0][2])
        meas[3] = float(personBox[0][3])

        #第一次检测
        if not found:
            for i in range(len(kf.errorCovPre)):
                kf.errorCovPre[i,i] = 1
            state[0] = meas[0]
            state[1] = meas[1]
            state[2] = 0
            state[3] = 0
            state[4] = meas[2]
            state[5] = meas[3]
            kf.statePost = state
            found = True

        else:
            kf.correct(meas) #Kalman修正
            
    videoWriter.write(res)