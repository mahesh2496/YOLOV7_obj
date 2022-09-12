import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import threading
from threading import Thread
import time
import cv2

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

conf_thres = 0.25

img_size = 640
iou_thres = 0.45


class Thread(threading.Thread):
    def __init__(self,previewName,ip):
        threading.Thread.__init__(self)
        #self.threadID = threadID
        self.previewName = previewName
        self.ip = ip

    def run(self):
        print("Start threadID" +str(self.previewName))
        self.detect()
        print("Exiting " + str(self.previewName))


    def detect(self):
        name = 'exp'
        trace = False
        project = 'runs/detect'
        save_conf = False
        save_txt = False
        device = ''
        view_img = False
        weights = 'D:\\people_2_09\\YOLOV7-OBJECT-COUNTER-main\\yolov7-main\\yolov7.pt'
        nosave='store_true'
        source = self.ip


        save_img = not nosave and not source.endswith('.txt')  # save inference images
        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))


        # Directories
        save_dir = Path(increment_path(Path(project) / name, exist_ok='store_true'))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        set_logging()
        device = select_device(device)
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(img_size, s=stride)  # check img_size

        if trace:
            model = TracedModel(model, device,img_size)

        if half:
            model.half()  # to FP16

        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

        # Set Dataloader
        vid_path, vid_writer = None, None
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        old_img_w = old_img_h = imgsz
        old_img_b = 1

        t0 = time.time()
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Warmup
            if device.type != 'cpu' and (
                    old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    model(img, augment=False)[0]

            pred = model(img, augment=False)[0]

            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)
            t3 = time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results\
                    obj_count = []

                    person_count = 0

                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or view_img:  # Add bbox to image

                            label = names[int(cls)]

                            if label == 'person':
                                obj_count.append(label)

                                person_count = obj_count.count('person')

                                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                                i = i + 1
                                # cv2.putText(im0, label, (0, 105), cv2.FONT_HERSHEY_TRIPLEX,
                                #             1, (0, 0, 0), 1)

                    if person_count > 0:
                        print(cv2.putText(im0, f"person_count:  {person_count}", (0, 100),
                                          cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2))

                im0 = cv2.resize(im0, (600, 600))
                cv2.imshow(str(self.previewName), im0)
                cv2.waitKey(1)  # 1 millisecond

        print(f'Done. ({time.time() - t0:.3f}s)')




if __name__ == '__main__':
    update = False
    rtsp_list = {
        1: "rtsp://admin:Admin123$@10.11.25.60:554/axis-media/media.amp",
        # 2: "rtsp://admin:Admin123$@10.11.25.64:554/axis-media/media.amp",
        # 3: "rtsp://admin:Admin123$@10.11.25.64:554/unicast/c2/s2/live",
        # 4: "rtsp://admin:Admin123$@10.11.25.60:554/media/video2",
        # 5: "rtsp://admin:Admin123$@10.11.25.59:554/Streaming/Channels/101",
        # 6:"rtsp://admin:Admin123$@10.11.25.64:554/axis-media/media.amp",
        # 7:"rtsp://admin:Admin123$@10.11.25.57:554/axis-media/media.amp",
        # 8: "rtsp://admin:Admin123$@10.11.25.65:554/axis-media/media.amp",
        # 9:"rtsp://admin:Admin123$@10.11.25.64:554/ch01/0"

    }

    threads = []
    for i, cam in rtsp_list.items():
        thread1 = Thread(i, cam)
        threads.append(thread1)
    for i in threads:
        i.start()




