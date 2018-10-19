---
layout: post
title:  "keras 模型的动态部署"
date:   2018-10-19 17:00:00

categories: keras
tags: 模型 部署 
author: "Victor"
---

aiServerAPI服务搭建

20181011_v1
20181019_v2
lida@youyuan.com

## 1.项目背景
Ai项目网站在整合模型的时候，变得繁重和缓慢，在生成环境和开发环境，都存在瓶颈问题，故将模型相关的服务器抽取成一个API服务，使用Django进行搭建。

## 2.项目环境
搭建服务器：192.168.16.36
项目路径：/opt/web/aiServer/

Python版本：python3.5

Django版本：django 2.0

## 3.项目搭建

cd /opt/web
```
python3 -m pip install virtualenv
virtualenv env_django2.0
source env_django2.0/bin/activate

python -m pip install django==2.0     1017
python3 -m pip install djangorestframework
python -m pip install django-filter
python -m pip install markdown

django-admin.py startproject aiServer
django-admin.py startapp yyai
python3 manage.py migrate

python3 manage.py createsuperuser --email admin@example.com --username admin
: 密码：  admin123
(env_django2.0) 192-168-16-36.youyuan-idc.com [/opt/web/aiServer] 2018-10-11 14:45:10
root@pts/6 # find .

.
./manage.py
./aiServer
./aiServer/wsgi.py
./aiServer/__init__.py
./aiServer/__init__.pyc
./aiServer/settings.pyc
./aiServer/__pycache__
./aiServer/__pycache__/__init__.cpython-35.pyc
./aiServer/__pycache__/settings.cpython-35.pyc
./aiServer/__pycache__/wsgi.cpython-35.pyc
./aiServer/__pycache__/urls.cpython-35.pyc
./aiServer/settings.py
./aiServer/urls.py
./yyai
./yyai/admin.py
./yyai/__init__.py
./yyai/tests.py
./yyai/models.py
./yyai/migrations
./yyai/migrations/__init__.py
./yyai/migrations/__pycache__
./yyai/migrations/__pycache__/__init__.cpython-35.pyc
./yyai/serializers.py       [下一节添加]
./yyai/__pycache__
./yyai/__pycache__/__init__.cpython-35.pyc
./yyai/__pycache__/models.cpython-35.pyc
./yyai/__pycache__/admin.cpython-35.pyc
./yyai/__pycache__/urls.cpython-35.pyc
./yyai/__pycache__/views.cpython-35.pyc
./yyai/__pycache__/serializers.cpython-35.pyc
./yyai/\
./yyai/urls.py
./yyai/views.py
./db.sqlite3
```

## 4.项目初级开发
### 4.1.基础设置及开发
cd /opt/web/aiServer/aiServer

vim settings.py
```python
ALLOWED_HOSTS = ['*']

INSTALLED_APPS = (
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'yyai',
)
```
vim urls.py
```python
from django.conf.urls import include, url
from django.contrib import admin
from rest_framework import routers
from yyai import views

router = routers.DefaultRouter()
router.register(r'users', views.UserViewSet)
router.register(r'groups', views.GroupViewSet)

urlpatterns = [
    # Examples:
    # url(r'^$', 'aiServer.views.home', name='home'),
    # url(r'^blog/', include('blog.urls')),

    #url(r'^admin/', include(admin.site.urls)),
    url(r'^', include(router.urls)),
    url(r'^api-auth/', include('rest_framework.urls', namespace='rest_framework'))
]
```

cd /opt/web/aiServer/yyai

vim serializers.py

```python
from django.contrib.auth.models import User, Group
from rest_framework import serializers


class UserSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = User
        fields = ('url', 'username', 'email', 'groups')


class GroupSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Group
        fields = ('url', 'name')
```

vim views.py

```python
from django.shortcuts import render
from django.contrib.auth.models import User, Group
from rest_framework import viewsets
from yyai import serializers #import UserSerializer, GroupSerializer

# Create your views here.
class UserViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows users to be viewed or edited.
    """
    queryset = User.objects.all().order_by('-date_joined')
    serializer_class = serializers.UserSerializer


class GroupViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows groups to be viewed or edited.
    """
    queryset = Group.objects.all()
    serializer_class = serializers.GroupSerializer
```

### 4.2.初步启动及测试
python manage.py runserver 0.0.0.0:8099

Web端访问：

命令行访问：
```
curl -H 'Accept: application/json; indent=4' -u admin:admin123 http://192.168.16.36:8099/users/
```

### 4.3.模型环境部署
```
python3 -m pip install numpy
python3 -m pip install keras
python3 -m pip install tensorflow==1.9
python3 -m pip install -U –pre numpy scipy matplotlib scikit-learn scikit-image
python3 -m pip install opencv_python
```
### 4.4.模型的加载和管理
vim urls.py

```python
from django.conf.urls import url
from yyai import views,facescore,predictsex

urlpatterns = [
    url(r'facescore/$', facescore.predict_face_score),
    url(r'facescore_loadmodel/$', facescore.loadModel),

    url(r'predictsex/$',predictsex.predict_sex),
    url(r'predictsex_loadmodel/$',predictsex.loadModel),
]
```

vim modelUtils.py

```python
import keras
from keras.models import load_model
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from scipy.misc import imresize
import numpy as np
import cv2
from keras.models import model_from_json
import matplotlib.pyplot as plt
import os
import logging
logger = logging.getLogger('django')

#使用H5文件导入模型，需要两个必要参数
#modelPath:'/model'
#testData=np.zeros((1,350,350,3),dtype=np.float32)
def upload_h5_Models(modelPath,testData):
    logger.debug("upload models begin")
    path_name=modelPath
    model = []
    modelList = []
    keras.backend.clear_session() #清理session反复识别注意
    #json_string=""
    #with open('./model/model_struct.json') as f:
    #    json_string=f.read()
#model_renew = model_from_json(json_string)

    for dir_item in os.listdir(path_name):
        #从初始路径开始叠加，合并成可识别的操作路径
        full_path = os.path.abspath(os.path.join(path_name, dir_item))
        logger.debug(full_path)
        if not full_path.endswith('.h5'):
            continue
        #model_renew.load_weights(full_path)

        model = [dir_item,load_model(full_path)]
        logger.debug("each load finish")

        logger.debug("%s" %model[1].predict_proba(testData))
        modelList.append(model)

    logger.debug("upload models end")
    return modelList
```

vim facescore.py

```python
from django.shortcuts import render
from django.contrib.auth.models import User, Group
from rest_framework import viewsets
from yyai import serializers #import UserSerializer, GroupSerializer
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import action
import numpy as np
from yyai import modelUtils

modelList = []

def loadModel(request):
    loadModelLocal()
    res = {"result":"load model success!"}
    return JsonResponse(res, safe=False)

def loadModelLocal():
    modelPath='/opt/web/aiServer/yyai/model/facescore/'
    testData=np.zeros((1,350,350,3),dtype=np.float32)
    modelList = modelUtils.upload_h5_Models(modelPath,testData)
    print(modelList)
    return modelList

# Create your views here.
@csrf_exempt
def predict_face_score(request):
res = {"name":"lida","path":"/home/lida"}
    print(modelList)
    return JsonResponse(res, safe=False)

modelList = loadModelLocal()
```

vim predictsex.py 

```python
from django.shortcuts import render
from django.contrib.auth.models import User, Group
from rest_framework import viewsets
from yyai import serializers #import UserSerializer, GroupSerializer
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import action
import numpy as np
from yyai import modelUtils

modelList = []

def loadModel(request):
    loadModelLocal()
    res = {"result":"load model success!"}
    return JsonResponse(res, safe=False)

def loadModelLocal():
    modelPath='/opt/web/aiServer/yyai/model/predictsex/'
    testData=np.zeros((1,350,350,3),dtype=np.float32)
    modelList = modelUtils.upload_h5_Models(modelPath,testData)
    print(modelList)
    return modelList

# Create your views here.
@csrf_exempt
def predict_sex(request):
res = {"name":"sex","path":"/home/lida"}
    print(modelList)
    return JsonResponse(res, safe=False)

modelList = loadModelLocal()
```

效果：项目启动，所有应用py都到各自设定的目录下加载模型，启动较慢.运行时，通过api可以出发不同的应用py更新自己的模型.

- http://192.168.16.36:8099/predictsex/
- http://192.168.16.36:8099/facescore/
- http://192.168.16.36:8099/predictsex_loadmodel/

### 4.5.模型部署问题点

- 问题点描述
  - keras.backend.clear_session() 导致后面模型加载时，前面的模型被clean；
  - 两个模型不能同时进行API的reload操作，因为他们使用相同的session，会报错；
  - 单个模型加载时，所有的模型操作都无法进行；
 
- 总之，以上的部署，在项目启动的时候，不会有问题，模型顺利加载。后面进行模型预测也没问题，但是一旦使用api 的模型reload操作，就会出现各种问题。不能进行模型的动态加载和多模型的并发使用。

- 归结到底，产生的问题是：
  - 模型不能动态加载
  - 模型需要重启服务
 
- 产生问题的主要原因是：
  - 模型都使用了tensorflow的同一个session和graph；
  - ad_model 和 predict不在同一个线程； （https://zhuanlan.zhihu.com/p/27101000）

## 5.项目的最终开发
### 5.1.Tensorflow的设置

TensorFlow 的特点：
- 图 (graph) 来表示计算任务.
- 称之为 会话 (Session) 的上下文 (context) 中执行图.
-  tensor 表示数据.
- 变量 (Variable) 维护状态.
-  feed 和 fetch 可以为任意的操作(arbitrary operation) 赋值或者从其中获取数据.

sess = tf.Session()
创建一个新的TensorFlow session。 
如果在构建session时没有指定graph参数，则将在session中启动默认关系图。如果使	用多个图（在同一个过程中使用tf.Graph()创建，则必须为每个图使用不同的sessio，但	是每个图都可以用于多个sessio中，在这种情况下，将图形显式地传递给sessio构造函	数通常更清晰。

sess.as_default()
返回使该对象成为默认session的上下文管理器。 

graph = tf.Graph()
graph.as_default()

** 在Tensorflow中，所有操作对象都包装到相应的Session中的，所以想要使用不同的模型就需要将这些模型加载到不同的Session中并在使用的时候申明是哪个Session，从而避免由于Session和想使用的模型不匹配导致的错误。而使用多个graph，就需要为每个graph使用不同的Session，但是每个graph也可以在多个Session中使用，这个时候就需要在每个Session使用的时候明确申明使用的graph。

```python
g1 = tf.Graph() # 加载到Session 1的graph
g2 = tf.Graph() # 加载到Session 2的graph

sess1 = tf.Session(graph=g1) # Session1
sess2 = tf.Session(graph=g2) # Session2
# 加载第一个模型with sess1.as_default(): 
    with g1.as_default():
        tf.global_variables_initializer().run()
        model_saver = tf.train.Saver(tf.global_variables())
        model_ckpt = tf.train.get_checkpoint_state(“model1/save/path”)
        model_saver.restore(sess, model_ckpt.model_checkpoint_path)# 加载第二个模型with sess2.as_default():  # 1
    with g2.as_default():  
        tf.global_variables_initializer().run()
        model_saver = tf.train.Saver(tf.global_variables())
        model_ckpt = tf.train.get_checkpoint_state(“model2/save/path”)
        model_saver.restore(sess, model_ckpt.model_checkpoint_path)

...
# 使用的时候with sess1.as_default():
    with sess1.graph.as_default():  # 2
        ...
with sess2.as_default():
    with sess2.graph.as_default():
        ...
# 关闭sess
sess1.close()
sess2.close()
```

注：
- 1、在1处使用as_default使session在离开的时候并不关闭，在后面可以继续使用直到手动关闭；
- 2、由于有多个graph，所以sess.graph与tf.get_default_value的值是不相等的，因此在进入sess的时候必须sess.graph.as_default()明确申明sess.graph为当前默认graph，否则就会报错。

设计上，为每个应用模型，分配一个session和graph，进行统一绑定管理。当使用特定模型的时候，采用如下模式：
```python
    with self.session.as_default():
        with self.graph.as_default():
            Use model do sth;
```

### 5.2.模型管理类

```python
import keras
from keras.models import load_model
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from scipy.misc import imresize
import numpy as np
import cv2
from keras.models import model_from_json
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import backend as K
import os
import logging
logger = logging.getLogger('django')

class ModelUtil(object):
    def __init__(self,modelPath,data):
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)
        self.modelList = []
        self.data = data
        self.modelPath = modelPath

    def __str__(self):
        return '(%s,%s,%s)' %(self.session, self.graph, self.modelList);

    def upload_h5_Models(self):
        logger.debug("upload models begin")
        logger.info(self.data.shape)
        path_name=self.modelPath
        self.modelList = []
        model = []
        with self.session.as_default():
            with self.graph.as_default():
                for dir_item in os.listdir(path_name):
                    #从初始路径开始叠加，合并成可识别的操作路径
                    full_path = os.path.abspath(os.path.join(path_name, dir_item))
                    logger.debug(full_path)
                    if not full_path.endswith('.h5'):
                        continue
                    model = [dir_item,load_model(full_path)]
                    logger.debug("each load finish")
                    logger.debug("%s" %model[1].predict(self.data))
                    self.modelList.append(model)
                logger.info(self.modelList)
            logger.debug("upload models end")

    def predict_proba(self,data):
        result = []
        with self.session.as_default():
            with self.graph.as_default():
                for m in self.modelList:
                    myObject= {}
                    logger.debug("predict use model %s" %(m[0]))
                    logger.debug(m[1])
                    #with graph.as_default():
                    myObject['name']=m[0]
                    proba = m[1].predict(data)
                    myObject['value']=[proba[0][0],proba[0][1]]
                    logger.debug("predict value is %s" %(myObject['value']))
                    result.append(myObject)
        return result

    def predict(self,data):
        result = []
        with self.session.as_default():
            with self.graph.as_default():
                for m in self.modelList:
                    myObject= {}
                    logger.debug("predict use model %s" %(m[0]))
                    logger.debug(m[1])
                    #with graph.as_default():
                    myObject['name']=m[0]
                    myObject['value']=round(m[1].predict(data)[0][0], 2)
                    logger.debug("predict value is %s" %(myObject['value']))
                    result.append(myObject)
        return result
```
        
### 5.3.应用api开发

```python
from django.shortcuts import render
from django.contrib.auth.models import User, Group
from rest_framework import viewsets
from yyai import serializers #import UserSerializer, GroupSerializer
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import action

from keras.preprocessing.image import array_to_img, img_to_array, load_img
from scipy.misc import imresize

import numpy as np
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from yyai.modelUtils import ModelUtil
#from yyai import modelUtils
import logging
logger = logging.getLogger('django')

################################################################
#模型路径需要修改
modelPath='/opt/web/aiServer/yyai/model/facescore/'
#模型输入数据实例
testData=np.zeros((1,350,350,3),dtype=np.float32)
#实例化模型管理类
modelU = ModelUtil(modelPath,testData)
logger.info(modelU)
modelU.upload_h5_Models()
logger.info(modelU)

#webapi接口，进行该应用的模型更新
def loadModel(request):
    logger.info(modelU)
    modelU.upload_h5_Models()
    logger.info(modelU)
    res = {"result":"load model success!"}
    return JsonResponse(res, safe=False)

def get_score(img):
    logger.info("enter in to get_score...")
    logger.info(modelU)
    img_height, img_width, channels = 350, 350, 3
    if modelU.modelList==[]:
        logger.info('no faceScore models ... return []')
        return {"result":"no faceScore models"}
    result = []
    try:
        resized_image = cv2.resize(img, (img_height, img_width))
    except :
        logger.info("resize is except: imresize" )
        return {"result":"resize is except"}
    test_x = resized_image
    test_x = test_x.astype("float32") / 255.
    test_x = test_x.reshape((1,) + test_x.shape)
    logger.info(test_x.shape)

    result = modelU.predict(test_x)

    return {"result":result}

# Create your views here.
# 对外的模型API接口
@csrf_exempt
def predict_face_score(request):
    res = {"name":"lida","path":"/home/lida"}
    #使用get或post模式都可以，yyImage使用POST模式
    if request.method == 'GET':
        logger.info("get data and return defalt")
        #name=request.GET.get("name")
        imgPath="uploads/admin/0_cropped_bdb19711-11b7-484e-b309-6f7c196687de.jpg"
        img = img_to_array(load_img(imgPath))
        print(img)
        res = "%s" %get_score(img)
    elif request.method == 'POST':
        #name=request.POST.get("name",'')
        imgPath = request.POST.get("imgPath",'')
        logger.info(imgPath)
        if imgPath=='':
            logger.info("no imgPath")
            res = [{'result':'no images'}]
        else:
            img = img_to_array(load_img(imgPath))
            res = "%s" %get_score(img)
    logger.info(res)
    return JsonResponse(res, safe=False)
```

### 4.5.后续模型添加
拷贝重写一个app的api接口python文件，同时配置url的访问映射，即可实现添加模型的、模型动态加载、模型API调用的功能。

## 6.总结
工程为机器学习的模型提供了一共独立的服务，该服务保证了模型的动态加载、并发更新、通过API接口提供对模型的调用，实现模型管理与前端项目的分离。
暂没有实现对计算资源的分布式设计，原因是当前生产环境还不具备分布式的需求和条件。

项目运行通过web端进行api接口调用的截图如下：






## Refer：
 - [1].https://www.w3cschool.cn/tensorflow_python/tensorflow_python-fibz28ss.html
 - [2].https://www.tensorflow.org/api_docs/python/tf/InteractiveSession
 - [3].https://www.cnblogs.com/arkenstone/p/7016481.html
 - [4].https://keras-cn-docs.readthedocs.io/zh_CN/latest/blog/keras_and_tensorflow/
 - [5].https://keras.io/zh/getting-started/faq/
 - [6].https://www.jianshu.com/p/76184d1a6358
 - [7].https://www.django-rest-framework.org/#
 - [8].https://www.tensorflow.org/programmers_guide/faq?hl=zh-cn
