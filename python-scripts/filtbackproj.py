import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageChops
from scipy.fftpack import fft, fftshift, ifft
import time

def dummyImg(size0, size1):
    """只是創建一個由實心矩形組成的虛擬8位元影像
    inputs: size0, size1 - dimensions of image in pixels
    output: dumImg - PIL image object"""
    M = np.zeros((size0, size1))
    a = round(size0/4)
    b = round(size1/4)
    # 在中心位置插入尺寸為影像大小 1/2 的矩形，使其居中。
    #M[a:size0-a,b:size1-b] = 255 
    M[190:210, :] = 255
    M[:, 210:230] = 255
    dumImg = Image.fromarray(M.astype('uint8'))  #create image object
    return dumImg

def padImage(img):
    """用零填滿影像，使新影像成為邊長等於原始影像對角線大小的正方形。
       傳回填滿後的影像以及植入原始影像的左上角座標"""
    N0, N1 = img.size
    lenDiag = int(np.ceil(np.sqrt(N0**2+N1**2)))
    imgPad = Image.new('L',(lenDiag, lenDiag))
    c0, c1 = int(round((lenDiag-N0)/2)), int(round((lenDiag-N1)/2)) 
    # coordinates of top-left corner in which to paste image
    imgPad.paste(img, (c0,c1)) 
    return imgPad, c0, c1

def getProj(img, theta):
    """將影像旋轉一系列角度並進行投影。請注意，我們預先填充圖像，而不是
       允許旋轉方法擴展圖像，因為圖像會根據旋轉角度擴展為不同的大小。
       我們需要將每個旋轉圖像擴展到相同的大小，以便可以將貢獻收集到同一向量中。
    inputs: img - PIL image object
            theta - 用於計算投影的一維 numpy 角度矩陣
    output: sinogram - n x m img在不同角度上的投影（實際上是radon transform)"""

    numAngles = len(theta)
    sinogram = np.zeros((img.size[0],numAngles)) # (y, x)

    #set up plotting
    plt.ion() # 打開交互模式，碰到plt.show()不會停止
    fig1, (ax1, ax2) = plt.subplots(1,2)
    im1 = ax1.imshow(img, cmap='gray')
    ax1.set_title('<-- Sum')
    im2 = ax2.imshow(sinogram, extent=[theta[0],theta[-1], img.size[0]-1, 0],
                     cmap='gray', aspect='auto')
    ax2.set_xlabel('Angle (deg)')
    ax2.set_title('Sinogram')
    plt.show()
    # plt.close()

    #get projections an dplot
    for n in range(numAngles): # 90 - theta[n]：對Y軸求和，順時針旋轉
        rotImgObj = img.rotate(90-theta[n], resample=Image.BICUBIC)
        im1.set_data(rotImgObj)
        sinogram[:,n] = np.sum(rotImgObj, axis=0) # 表示沿着垂直的方向求和 
        # im2.set_data將圖片更新為新的資料，np.ptp求最大值與最小值之差值
        im2.set_data(Image.fromarray((sinogram-np.min(sinogram))/np.ptp(sinogram)*255))
        fig1.canvas.draw() # 用來重新繪製整張圖表
        fig1.canvas.flush_events() # 用於在每次迭代時清除圖形，以使後續圖形不會重疊。

    plt.ioff() # 顯示前關掉交互模式
    return sinogram

def projFilter(sino):
    """過濾投影：通常，在濾波反投影中使用Ramp濾波器乘以window函數。這裡的濾波器
       函數可以透過單一參數「a」進行調整，以近似純Ramp濾波器（a ~ 0）或乘以具有
       增加截止頻率的sinc視窗（a ~ 1）。以上為Wakas Aqram的貢獻。 
    inputs: sino - [n x m] numpy array，n是投影數量，m是使用的角度數量。
    outputs: filtSino - [n x m] filtered sinogram array"""
    
    a = 0.1;
    projLen, numAngles = sino.shape
    step = 2*np.pi/projLen
    w = arange2(-np.pi, np.pi, step)
    if len(w) < projLen:
    # 根據圖像大小，可能是 len(w) = projLen - 1。 
    # 在這種情況下，另一個元素被加到 w
        w = np.concatenate([w, [w[-1]+step]]) 
    # 帶有函數abs(sin(w))的Ramp濾波器abs(w)的近似
    rn1 = abs(2/a*np.sin(a*w/2));  
    rn2 = np.sin(a*w/2)/(a*w/2); #sinc window with 'a' modifying the cutoff freqs
    r = rn1*(rn2)**2;            #modulation of ramp filter with sinc window
    
    filt = fftshift(r)  # 將零頻率分量移至頻譜中心。 
    #filt = 1
    filtSino = np.zeros((projLen, numAngles))
    for i in range(numAngles):
        projfft = fft(sino[:,i]) # 將某一角度投影做FFT轉換
        filtProj = projfft*filt  # 
        filtSino[:,i] = np.real(ifft(filtProj))

    return filtSino
        
def backproject(sinogram, theta):
    """Backprojection function. 
    inputs:  sinogram - [n x m] numpy array，n是投影數量，m是使用的角度數量。
             theta - 長度為 m 的向量表示正弦圖中表示的角度
    output: backprojArray - [n x n] backprojected 2-D numpy array"""
    imageLen = sinogram.shape[0]
    reconMatrix = np.zeros((imageLen, imageLen))
    #create coordinate system centered at (x,y = 0,0)
    x = np.arange(imageLen)-imageLen/2 
    y = x.copy()
    X, Y = np.meshgrid(x, y)

    plt.ion()
    fig2, ax = plt.subplots()
    im = plt.imshow(reconMatrix, cmap='gray')

    theta = theta*np.pi/180
    numAngles = len(theta)

    for n in range(numAngles):
        # 確定繞原點旋轉的 x 座標，以網格形式表示
        Xrot = X*np.sin(theta[n])-Y*np.cos(theta[n]) # X, Y, Xrot皆二維矩陣
        # 切換回原始影像座標，將數值四捨五入以生成索引
        XrotCor = np.round(Xrot+imageLen/2) 
        XrotCor = XrotCor.astype('int')
        projMatrix = np.zeros((imageLen, imageLen))
        # 旋轉後，不可避免地會得到超出原始大小的新座標
        # np.where用於查找滿足條件的元素的索引，返回元素的索引元組(x, y)。
        mx, my = np.where((XrotCor >= 0) & (XrotCor <= (imageLen-1))) 
        s = sinogram[:,n] # 獲取第n個角度的投影數據
        # 將投影數據映射到反投影矩陣的合理位置，backproject in-bounds data
        projMatrix[mx, my] = s[XrotCor[mx, my]]  
        reconMatrix += projMatrix
        im.set_data(Image.fromarray((reconMatrix-np.min(reconMatrix))/np.ptp(reconMatrix)*255))
        ax.set_title('Theta = %.2f degrees' % (theta[n]*180/np.pi))
        fig2.canvas.draw()
        fig2.canvas.flush_events()
         
    plt.close()
    plt.ioff()
    backprojArray = np.flipud(reconMatrix)
    return backprojArray

def arange2(start, stop=None, step=1):
    """#Modified version of numpy.arange which corrects error associated with non-integer step size"""
    if stop == None:
        a = np.arange(start)
    else: 
        a = np.arange(start, stop, step)
        if a[-1] > stop-step:   
            a = np.delete(a, -1)
    return a

#def main():
if __name__ == '__main__':
    start_time = time.perf_counter()
    #myImg = dummyImg(400,400)
    # convert('L')轉成灰階影像，每個像素以8個bit表示，0表示黑，255表示白。
    # myImg = Image.open('SheppLogan.png').convert('L')
    myImg = Image.open('Python scripts/OIP.png').convert('L')
        
    myImgPad, c0, c1 = padImage(myImg)  #PIL image object
    dTheta = 1
    theta = np.arange(0,361,dTheta)
    print('Getting projections\n')
    mySino = getProj(myImgPad, theta)  #numpy array
    print('Filtering\n')
    filtSino = projFilter(mySino)  #numpy array
    print('Performing backprojection')  

    recon = backproject(filtSino, theta)
    recon2 = np.round((recon-np.min(recon))/np.ptp(recon)*255) #convert values to integers 0-255
    reconImg = Image.fromarray(recon2.astype('uint8'))
    n0, n1 = myImg.size
    reconImg = reconImg.crop((c0, c1, c0+n0, c1+n1))

    end_time = time.perf_counter()
    print(f"Execution time: {end_time - start_time:.6f} seconds")

    fig3, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(12,4))
    ax1.imshow(myImg, cmap='gray')
    ax1.set_title('Original Image')
    ax2.imshow(reconImg, cmap='gray')
    ax2.set_title('Filtered Backprojected Image')
    ax3.imshow(ImageChops.difference(myImg, reconImg), cmap='gray') #note this currently doesn't work for imported images
    ax3.set_title('Error')
    plt.show()
    # plt.close()
