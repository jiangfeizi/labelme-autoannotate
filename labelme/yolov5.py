from cv2 import merge
import labelme.inference
requirements = ['opencv-python', 'onnxruntime', 'numpy']
labelme.inference.Infer.install_requirements(requirements)

import numpy as np
import cv2
import onnxruntime

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def box_iou(box1, box2):
    lt = np.maximum(box1[:2], box2[:2])
    rb = np.minimum(box1[2:4], box2[2:4])

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])


    if lt[0] > rb[0] or lt[1] > rb[1]:
        return 0.0

    interBoxS = (rb[0] - lt[0]) * (rb[1] - lt[1])
    return interBoxS / (box1_area + box2_area - interBoxS)

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, max_det=300):
    output = []
    max_nms = 30000

    prediction = prediction[0]     
    prediction[:, 5:] *= prediction[:, 4:5]
    score_best = np.max(prediction[:, 5:], axis=1)
    xc = score_best > conf_thres
    x = prediction[xc]

    n = x.shape[0]  # number of boxes
    if not n:  # no boxes
        return output

    score_best = np.max(x[:, 5:], axis=1)
    x = x[np.argsort(-score_best)]
    arg_best = np.argmax(x[:, 5:], axis=1)

    if n > max_nms:  # excess boxes
        x = x[:max_nms]  # sort by confidence

    # Batched NMS
    boxes = xywh2xyxy(x[:, :4]).tolist()
    class_best = arg_best.tolist()

    while boxes:
        box1 = boxes.pop(0)
        cl = class_best.pop(0)
        output.append((box1, cl))

        pop_indexes = []
        for index, box2 in enumerate(boxes):
            if box_iou(box1, box2) > iou_thres:
                pop_indexes.append(index)
        for index in pop_indexes[::-1]:
            boxes.pop(index)
            class_best.pop(index)

    return output

class Infer(labelme.inference.Infer):
    def __init__(self) -> None:
        super().__init__()
        self.weight = r'D:\Code\github\yolov5\yolov5s.onnx'
        self.classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']
        cuda = False
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        self.session = onnxruntime.InferenceSession(self.weight, providers=providers)

    def predict(self, image, scale) -> list:
        shapes = []
        image, ratio, (dw, dh) = letterbox(image, (scale, scale), (0, 0, 0), False)
        image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        image = np.ascontiguousarray(image)
        image = np.array(image, dtype=np.float32) / 255
        if len(image.shape) == 3:
            image = image[None]  # expand for batch dim
        prediction = self.session.run([self.session.get_outputs()[0].name], {self.session.get_inputs()[0].name: image})[0]

        output = non_max_suppression(prediction, 0.25, 0.05)
        for box, cl in output:
            box[0] = (box[0] -dw) / ratio[0]
            box[1] = (box[1] -dh) / ratio[0]
            box[2] = (box[2] -dw) / ratio[0]
            box[3] = (box[3] -dh) / ratio[0]
            shapes.append(self.get_shape(self.classes[cl] if self.classes else str(cl), [[box[0], box[1]], [box[2], box[3]]], 'rectangle'))
        return shapes
