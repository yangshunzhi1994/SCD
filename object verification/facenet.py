import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from nets.facenet import Facenet as facenet
from utils.utils import preprocess_input, resize_image, show_config



class Facenet(object):
    _defaults = {
        "model_path"    : "model_data/facenet_mobilenet.pth",
        #   输入图片的大小。
        "input_shape": [160, 160, 3],
        "backbone": "mobilenet",
        "letterbox_image": True,
        "cuda": True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        self.generate()
        
        show_config(**self._defaults)
        
    def generate(self):
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = facenet(backbone=self.backbone, mode="predict").eval()
        self.net.load_state_dict(torch.load(self.model_path, map_location=device), strict=False)

        if self.cuda:
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True
            self.net = self.net.cuda()

    def detect_image(self, image_1, image_2):
        with torch.no_grad():
            image_1 = resize_image(image_1, [self.input_shape[1], self.input_shape[0]], letterbox_image=self.letterbox_image)
            image_2 = resize_image(image_2, [self.input_shape[1], self.input_shape[0]], letterbox_image=self.letterbox_image)
            
            photo_1 = torch.from_numpy(np.expand_dims(np.transpose(preprocess_input(np.array(image_1, np.float32)), (2, 0, 1)), 0))
            photo_2 = torch.from_numpy(np.expand_dims(np.transpose(preprocess_input(np.array(image_2, np.float32)), (2, 0, 1)), 0))
            
            if self.cuda:
                photo_1 = photo_1.cuda()
                photo_2 = photo_2.cuda()

            output1 = self.net(photo_1).cpu().numpy()
            output2 = self.net(photo_2).cpu().numpy()
            
            #---------------------------------------------------#
            #   计算二者之间的距离
            #---------------------------------------------------#
            l1 = np.linalg.norm(output1 - output2, axis=1)

        return l1


if __name__ == "__main__":
    model = Facenet()

    while True:
        image_1 = input('Input image_1 filename:')
        try:
            image_1 = Image.open(image_1)
        except:
            print('Image_1 Open Error! Try again!')
            continue

        image_2 = input('Input image_2 filename:')
        try:
            image_2 = Image.open(image_2)
        except:
            print('Image_2 Open Error! Try again!')
            continue

        probability = model.detect_image(image_1, image_2)
        print(probability)
