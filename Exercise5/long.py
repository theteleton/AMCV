import torch
import numpy as np
import cv2
from got10k.trackers import Tracker
from siamfc.siamfc import TrackerSiamFC
from siamfc.ops import crop_and_resize

class LongSiamFC(Tracker):
    def __init__(self, net_path, threshold, sampling_method, sigma, n, **kwargs):
        super(LongSiamFC, self).__init__("LongSiamFC", True)
        self.short = TrackerSiamFC(net_path=net_path)
        self.threshold = threshold
        self.sampling_method = sampling_method
        self.sigma = sigma
        self.n = n

        
    def init(self, image, box):
        self.short.init(image, box)
        self.position, self.original_response = self.short.update(image)



    @torch.no_grad()
    def update(self, image):
        position, response = self.short.update(image)
        confidence_score = response / self.original_response
        
        if confidence_score >= self.threshold:
            self.last_good_center = self.short.center
            self.last_good_size = self.short.target_sz
            self.lost = 0
            return position, response
        else:
            h = image.shape[0]
            w = image.shape[1]
            x = self.last_good_center[1]
            y = self.last_good_center[0]

            sizex = self.last_good_size[1]
            sizey = self.last_good_size[0]
            samplex = []
            sampley = []
            if self.sampling_method == "uniform":
                samplex = np.random.randint(sizex + 1, w - sizex - 1, self.n)
                sampley = np.random.randint(sizey + 1, h - sizey - 1, self.n)
            elif self.sampling_method == "gauss":
                samplex = np.random.normal(x, self.sigma * min(sizex, sizey), self.n)
                sampley = np.random.normal(y, self.sigma * min(sizex, sizey), self.n)
            else:
                samplex = np.random.normal(x, self.sigma * min(sizex, sizey) * self.lost, self.n)
                sampley = np.random.normal(y, self.sigma * min(sizex, sizey) * self.lost, self.n)
                samplex = [min(xi, w - sizex - 1) for xi in samplex]
                sampley = [min(yi, h - sizey - 1) for yi in sampley]
                samplex = [max(xi, sizex + 1) for xi in samplex]
                sampley = [max(yi, sizey + 1) for yi in sampley]

            samples = [(int(x), int(y)) for (x, y) in zip(samplex, sampley)]

            self.short.net.eval()

            images = [image[int(y - sizey / 2) : int(y + sizey / 2), int(x - sizex / 2) : int(x + sizex / 2)] for (x, y) in samples]
            print(images)
            print(self.short.cfg.instance_sz)
            x = [cv2.resize(
                img, [self.short.cfg.instance_sz, self.short.cfg.instance_sz]) for img in images]
            x = np.stack(x, axis=0)
            x = torch.from_numpy(x).to(
                self.short.device).permute(0, 3, 1, 2).float()

            x = self.short.net.backbone(x)
            responses = self.short.net.head(self.short.kernel, x)
            responses = responses.squeeze(1).cpu().numpy()

            idx = np.argmax(np.amax(responses, axis=(1, 2)))

            max_response = np.max(responses[idx])

            self.short.center = np.array([samples[idx][0], samples[idx][1]]).astype("float64")
            self.short.target_sz = self.last_good_size
            
            position, response = self.short.update(image)
            confidence_new = response / self.original_response

            if confidence_new > self.threshold:
                self.lost = 0
                self.last_good_center = self.short.center
                self.last_good_size = self.short.target_sz
                return position, response
            else:
                self.lost = self.lost + 1
                return position, 0



    
