import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import tifffile
import time

#这3个都是单声道文件
path1 = 'wav/1、女声a第二声-0.625s-sr=16000.wav'
path2 = 'wav/2、女声中文朗读-9s-sr=16000.wav'
path3 = 'wav/3、歌曲大海-张雨生-20s-sr=44100.wav'

#准备信号
path = path1
signal, sr = sf.read(path, dtype='float32')#sr是采样率
signal = signal*1e8#将信号放大，从而减小A1'/A1
signal_len = signal.shape[0]#信号长度
delta_t = 1/sr#Δt，点间距表示的时长
timeline = np.arange(0,signal_len/sr,delta_t, dtype='float32')#时间轴
print('signal.shape=',signal.shape)
print('sr=',sr)

#信号的声波图
plt.figure(figsize=(16,4))
plt.scatter(timeline,signal,s=0.1) 
plt.show()

#系统参数设置
f0 = np.arange(20, 3020, 10, dtype='float32')
b = 3
m = 1/(2*np.pi)**2
w0 = 3*np.pi*f0
k = m*w0**2
y = b/(2*m)
w = np.sqrt(w0**2 - y**2)
f_num = f0.shape[0]#系统的个数
print('f_num=',f_num)

#迭代计算
coswt = np.cos(w*delta_t)
sinwt = np.sin(w*delta_t)
envelope = np.exp(-y*delta_t)
def cal_ite(x0, v0, F0, F1):
    c = (F1-F0)/delta_t
    r = c/k
    s = F0/k-(b*c)/(k**2)
    Asina = x0-s
    Acosa = (v0-r+y*Asina)/w
    Asinwta = Acosa*sinwt + Asina*coswt
    Acoswta = Acosa*coswt - Asina*sinwt
    x = envelope*Asinwta + r*delta_t + s
    v = envelope*(-y*Asinwta+w*Acoswta) + r
    return x, v

x = np.zeros((f_num, signal_len), dtype='float32')
v = np.zeros((f_num, signal_len), dtype='float32')
time1 = time.time()
for i in range(1, signal_len, 1):
    x[:,i], v[:,i] = cal_ite(x[:,i-1], v[:,i-1], signal[i-1], signal[i])
print('耗时',str(time.time()-time1))


#方法：压缩响应的长度，仅供参考
#输入的x已经取了绝对值
#对每个响应，先按频率确定窗口大小，一个窗口对应一个周期。令窗口内所有点的值都等于这个窗口内的最大值。
#至于末尾的一小截，直接都改为前一个窗口内点的值。至此，响应的长度未变。
#再对所有响应，按统一的窗口大小，取窗口内的平均值。至此，响应的长度=窗口数
def compress(x,sr,signal_len,f0,f_num,avg_win_len):
    win_len = np.ceil(sr/f0).astype('int32')#平均每个周期多少个点，向上取整
    frame_num = np.floor(signal_len/win_len).astype('int32')#整个信号有多少个窗口，向下取整
    temp_len = frame_num*win_len#所有完整窗口加起来的长度
    temp = None
    for i in range(0,f_num,1):
        temp = x[i,0:temp_len[i]].reshape(frame_num[i],win_len[i])
        temp = np.max(temp,axis=1)#每个窗口内取最大值，结果的shape为(窗口数,)
        temp = temp.reshape(frame_num[i],1)
        temp = np.tile(temp,(1,win_len[i]))#沿1轴复制，次数为窗口长度
        temp = temp.reshape(-1)#展开为1维
        x[i,0:temp.shape[0]] = temp
        #末尾一小截，都改为temp的最后一个点的值
        x[i,temp_len[i]:] = temp[temp.shape[0]-1:temp.shape[0]]

    #要压缩为声谱图，窗口大小、数量都是统一的，和频率无关
    win_len = avg_win_len
    frame_num = int(signal_len/win_len)#窗口数，向下取整
    x = x[:,0:frame_num*win_len].reshape(f_num,frame_num,win_len)
    x = np.average(x,axis=2)#每个频率的每个窗口内取最大值，结果的shape为(频率数,窗口数)

    #如果末尾还有一小截，仅提示，不做处理
    if win_len*frame_num<signal_len:
        print('末尾还有一小截')

    return x

x = x*(b*w0).reshape(f_num,1)#将响应放大
x = abs(x)
avg_win_len=50#窗口长度
x = compress(x,sr,signal_len,f0,f_num,avg_win_len)

x = x/np.max(x)
x = np.log(1+x)
x = x/np.log(2)

print('x.shape=',x.shape)

#显示压缩后的响应，即声谱图
plt.figure(figsize=(4,6))
plt.imshow(np.flip(x,axis=0))#上下翻转，让图片从下到上是低频到高频
plt.axis('off')#不显示坐标轴
plt.show()

#保存声谱图为.tiff文件
file_name = str(time.time())
path_result = 'tiff/resonance'+file_name+'.tiff'
tifffile.imsave(path_result, np.flip(x,axis=0))#1是白色，0是黑色。flip是上下翻转
print('——done——')
