
import cv2,os,glob,itertools,tqdm
from random import shuffle
from keras.preprocessing.image import img_to_array, ImageDataGenerator, load_img
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.applications.densenet import preprocess_input


class Load_Data(object):

    def __init__(self, config):
        self.train_data_path = config.train_data_path
        self.normal_size = config.normal_size
        self.channles = config.channles
        self.classNumber = config.classNumber
        self.cut = config.cut
        self.config = config


    def get_file(self,path):
        ends = os.listdir(path)[0].split('.')[-1]
        img_list = glob.glob(os.path.join(path , '*.'+ends))
        return img_list

    def load_data(self):
        categories = list(map(self.get_file, list(map(lambda x: os.path.join(self.train_data_path, x), os.listdir(self.train_data_path)))))
        data_list = list(itertools.chain.from_iterable(categories))
        shuffle(data_list)
        images_data ,labels_idx, labels = [], [], []

        with_platform = os.name

        for file in tqdm.tqdm(data_list):
            # if self.channles == 3:
            #     img = cv2.imread(file)
            #     print('resize之前的图片尺寸是:', img.shape)
            #     # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            #     # img = cv2.threshold(img,128,255,cv2.THRESH_BINARY)[-1]
            #     _, w, h = img.shape[::-1]
            # elif self.channles == 1:
            #     # img=cv2.threshold(cv2.imread(file,0), 128, 255, cv2.THRESH_BINARY)[-1]
            #     img = cv2.imread(file, 0)
            #     # img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)[-1]
            #     w, h = img.shape[::-1]
            print('file:', file)
            img = load_img(path=file, target_size=(224, 224), color_mode='rgb', interpolation='nearest')

            if with_platform == 'posix':  # 对应windows操作系统
                label = file.split('/')[-2]
            elif with_platform == 'nt':  # 对应windows操作系统
                label = file.split('\\')[-2]

            # print('img:',file,' has label:',label)
            img = img_to_array(img, 'channels_last')
            img = np.array([img])/255.0
            test_datagen = ImageDataGenerator(
                preprocessing_function=preprocess_input,
                # rotation_range=30,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True
            )
            img_array = test_datagen.flow(img, batch_size=16)
            for i in img_array:
                img = i[0]
                break
            print('type:', img)
            print('img:', img)

            images_data.append(img)
            labels.append(label)

        with open('train_class_idx.txt', 'r') as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
            for label in labels:
                idx = lines.index(label.rstrip())
                labels_idx.append(idx)

        # images_data = np.array(images_data,dtype='float')/255.0
        images_data = np.array(images_data, dtype='float32') / 255.0
        labels = to_categorical(np.array(labels_idx), num_classes=self.classNumber)
        X_train, X_test, y_train, y_test = train_test_split(images_data, labels)
        return X_train, X_test, y_train, y_test