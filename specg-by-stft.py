import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
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

#短时傅里叶变换
n_fft=512
hop_length=64
D = librosa.stft(y=signal, n_fft=n_fft, hop_length=hop_length, window='hann')
frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
print('D.shape=',D.shape)
print('frequencies.shape=',frequencies.shape)
#print('frequencies=',frequencies)
print('频率间隔=',frequencies[1]-frequencies[0])

x = np.abs(D)#振幅
x = x/np.max(x)
x = np.log(1+x)
x = x/np.log(2)

#显示声谱图
plt.figure(figsize=(4,6))
plt.imshow(np.flip(x,axis=0))#上下翻转，让图片从下到上是低频到高频
plt.axis('off')#不显示坐标轴
plt.show()

#保存声谱图为.tiff文件
file_name = str(time.time())
path_result = 'tiff/stft'+file_name+'.tiff'
tifffile.imsave(path_result, np.flip(x,axis=0))#1是白色，0是黑色。flip是上下翻转
print('——done——')
