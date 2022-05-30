import PIL
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode


class Segmentator:

    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device

    def __call__(self, input):
        x = self.preprocess(input).to(device=self.device)
        y = self.model(x)['out']
        segmented, categories = self.postprocess(y)
        return segmented, categories

    def segment_depth(self, categories, depth_image, label):
        segmented_depth = np.where(
            (categories == label),
            depth_image, np.zeros(depth_image.shape)
        ).astype(np.float32)

        return segmented_depth

    @classmethod
    def preprocess(self, frame):
        trf = T.Compose([
                        T.ToTensor(),
                        T.Resize((192, 256), InterpolationMode.BILINEAR),
                        T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])

        # frame = PIL.Image.fromarray(frame)
        return trf(frame).unsqueeze(0)

    @classmethod
    def postprocess(self, y, height=480, width=640):
        label_colors = np.array([(0, 0, 0),  # 0=background
                                 # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                                 (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                                 # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                                 (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                                 # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                                 (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                                 # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                                 (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

        # t = 1
        # y = torch.softmax(y/t, dim=1)
        # y = y[:, 1, ...]
        # idx = torch.rand_like(y)
        # category_map = torch.zeros_like(y).int()
        # category_map[y > 0.2] = 1
        # category_map = category_map.squeeze().cpu().numpy()
        category_map = torch.argmax(y.squeeze(), dim=0).detach().cpu().numpy()

        r = np.zeros_like(category_map).astype(np.uint8)
        g = np.zeros_like(category_map).astype(np.uint8)
        b = np.zeros_like(category_map).astype(np.uint8)

        for l in range(len(label_colors)):
            idx = category_map == l
            r[idx] = label_colors[l, 0]
            g[idx] = label_colors[l, 1]
            b[idx] = label_colors[l, 2]

        segmented_map = np.stack([r, g, b], axis=2)
        segmented_map = np.array(T.Resize((height, width), interpolation=InterpolationMode.NEAREST)(T.ToPILImage()(segmented_map)))

        category_map = np.array(T.Resize((height, width), interpolation=InterpolationMode.NEAREST)(
            T.ToPILImage()(category_map.astype(np.float32)))).astype(int)

        return segmented_map, category_map
