# YOLO-透過darknet.py進行影像串流
###### tags: `YOLO`

>YOLO有darknet.py，但只能讀取單張照片。因此紀錄如何使用修改darknet.py、image.c、image.h、Makefile，將YOLO的及時影像串流的辨識結果進行讀取，並做其他應用。

>本環境為Python3.6，於Ubuntu1804

1. 下載pjreddie的YOLO，並進行編譯 (make後產生libdarknet.so)
```bash=
git clone https://github.com/pjreddie/darknet.git
cd darknet
make
```
2. darknet.py是依賴libdarknet.so這份文件，所以要**將darknet.py移動到與libdarknet.so相同的位置**(darknet下)，並將darknet.py裡的CDLL路徑更改成libdarknet.so新路徑
```bash=
lib = CDLL("/home/jim/Desktop/darknet_pjreddie/libdarknet.so", RTLD_GLOBAL)
```
3. 處理darknet的錯誤程式碼 
- 由於以下程式碼會將這些位置參數傳至libdarknet.so，而libdarknet.so是用C/C++撰寫，因此會出現錯誤。
解決方法: 開啟darknet.py後，將路徑前面加上b，
```
if __name__ == "__main__":
     net = load_net(b"cfg/tiny-yolo.cfg", b"tiny-yolo.weights", 0)
    meta = load_meta(b"cfg/coco.data")
    r = detect(net, meta, b"data/dog.jpg")
    print r
```
4. 將darknet.py內自定義一個函數，透過此函數將影像傳至image.c，並讀取image.c的回傳辨識結果
```bash=
def nparray_to_image(img):
    data = img.ctypes.data_as(POINTER(c_ubyte))
    image = ndarray_image(data, img.ctypes.shape, img.ctypes.strides)
    return image
```

5. 修改darknet.py中detect函數的前幾行，前幾行更改後為:
```python=
def detect(net, meta, im, thresh=.5, hier_thresh=.5, nms=.45):
    #im = load_image(image, 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    ...
```
此處將image修改成im。image是字串型態，即路徑，並調用load_meta函數後得到im(一個darknet自定義的image類型)，再進行處理。而此處打算直接讀入影像進行辨識，因此省去load_meta函數。


6. 將darknet.py內加入以下程式碼，調用image.c的涵式
```bash=
ndarray_image = lib.ndarray_to_image
ndarray_image.argtypes = [POINTER(c_ubyte), POINTER(c_long), POINTER(c_long)]
ndarray_image.restype = IMAGE
```
7. 在src/image.c中，定義以下涵式
```bash=
#ifdef NUMPY
image ndarray_to_image(unsigned char* src, long* shape, long* strides)
{
    int h = shape[0];
    int w = shape[1];
    int c = shape[2];
    int step_h = strides[0];
    int step_w = strides[1];
    int step_c = strides[2];
    image im = make_image(w, h, c);
    int i, j, k;
    int index1, index2 = 0;

    for(i = 0; i < h; ++i){
            for(k= 0; k < c; ++k){
                for(j = 0; j < w; ++j){

                    index1 = k*w*h + i*w + j;
                    index2 = step_h*i + step_w*j + step_c*k;
                    //fprintf(stderr, "w=%d h=%d c=%d step_w=%d step_h=%d step_c=%d \n", w, h, c, step_w, step_h, step_c);
                    //fprintf(stderr, "im.data[%d]=%u data[%d]=%f \n", index1, src[index2], index2, src[index2]/255.);
                    im.data[index1] = src[index2]/255.;
                }
            }
        }

    rgbgr_image(im);

    return im;
}
#endif
```
8. 在image.h內宣告涵式
```bash=
#ifdef NUMPY
image ndarray_to_image(unsigned char* src, long* shape, long* strides);
#endif
```
9. 在Makefile內加入以下程式碼 (不確定是否要將python2.7改成自己環境的版本)
```
ifeq ($(NUMPY), 1) 
COMMON+= -DNUMPY -I/usr/include/python2.7/ -I/usr/lib/python2.7/dist-packages/numpy/core/include/numpy/
CFLAGS+= -DNUMPY
endif
```
並在上方加入NUMPY = 1 ，讓image.c中的 #ifdef NUMPY 成立
加入後會像這樣:

```
GPU=1
CUDNN=1
OPENCV=1
OPENMP=0
NUMPY=1
DEBUG=0
```
9. 重新編譯
```bash=
make clean
make
```

10. 在darknet.py增加讀取影像以及辨識功能
回傳的r有**類別**以及**bounding box**
```python=
if __name__ == "__main__":
    net = load_net(b"cfg/yolov3.cfg", b"yolov3.weights", 0)
    meta = load_meta(b"cfg/coco.data")
    vid = cv2.VideoCapture(0)
    while True:
        return_value,arr=vid.read()
        im=nparray_to_image(arr)
        r = detect(net, meta, im)
        print(r) 
```
---
# 番外
:bulb:可使用我的Sample.py，進行影像串流至網頁，檔案說明可參考 ➜[這篇](https://hackmd.io/4iTpehQkRq6iE3I5q3gEZg?view)

```bash=
git clone https://github.com/jim93073/yolo_socket.git
```
並將檔案移動至darknet底下

---
Reference ➜ [YOLO實戰](http://www.jeepxie.net/article/319977.html)
Reference ➜[github issue](https://github.com/pjreddie/darknet/issues/289#issuecomment-342448358)
