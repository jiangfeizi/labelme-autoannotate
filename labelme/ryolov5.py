from cv2 import merge
import labelme.inference
requirements = ['opencv-python', 'onnxruntime', 'numpy', 'shapely']
labelme.inference.Infer.install_requirements(requirements)

import numpy as np
import cv2
import onnxruntime
import shapely
import shapely.geometry

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

def xywhrm2xyxyxyxy(xywhrm):
    """
        xywhrm : shape (N, 6)
        Transform x,y,w,h,re,im to x1,y1,x2,y2,x3,y3,x4,y4
        Suitable for both pixel-level and normalized
    """
    x0, x1, y0, y1 = -xywhrm[:, 2:3]/2, xywhrm[:, 2:3]/2, -xywhrm[:, 3:4]/2, xywhrm[:, 3:4]/2
    xyxyxyxy = np.concatenate((x0, y0, x1, y0, x1, y1, x0, y1), axis=-1).reshape(-1, 4, 2)
    R = np.zeros((xyxyxyxy.shape[0], 2, 2), dtype=xyxyxyxy.dtype)
    R[:, 0, 0], R[:, 1, 1] = xywhrm[:, 4], xywhrm[:, 4]
    R[:, 0, 1], R[:, 1, 0] = xywhrm[:, 5], -xywhrm[:, 5]
    
    xyxyxyxy = np.matmul(xyxyxyxy, R).reshape(-1, 8)+xywhrm[:, [0, 1, 0, 1, 0, 1, 0, 1]]
    return xyxyxyxy

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, max_det=300):
    output = []
    max_nms = 30000

    prediction = prediction[0]     
    prediction[:, 7:] *= prediction[:, 6:7]
    score_best = np.max(prediction[:, 7:], axis=1)
    xc = score_best > conf_thres
    x = prediction[xc]

    n = x.shape[0]  # number of boxes
    if not n:  # no boxes
        return output

    score_best = np.max(x[:, 7:], axis=1)
    x = x[np.argsort(-score_best)]
    arg_best = np.argmax(x[:, 7:], axis=1)

    if n > max_nms:  # excess boxes
        x = x[:max_nms]  # sort by confidence

    # Batched NMS
    boxes = xywhrm2xyxyxyxy(x[:, :6]).tolist()
    class_best = arg_best.tolist()

    while boxes:
        box1 = boxes.pop(0)
        cl = class_best.pop(0)
        output.append((box1, cl))

        pop_indexes = []
        for index, box2 in enumerate(boxes):
            if polygon_inter_union_cpu(box1, box2) > iou_thres:
                pop_indexes.append(index)
        for index in pop_indexes[::-1]:
            boxes.pop(index)
            class_best.pop(index)

    return output

def polygon_inter_union_cpu(box1, box2):
    polygon1 = shapely.geometry.Polygon(np.array(box1).reshape(4,2)).convex_hull
    polygon2 = shapely.geometry.Polygon(np.array(box2).reshape(4,2)).convex_hull
    inter = polygon1.intersection(polygon2).area
    union = polygon1.union(polygon2).area
    return inter/union


class Infer(labelme.inference.Infer):
    def __init__(self) -> None:
        super().__init__()
        self.weight = r'C:\Users\A5324\Desktop\rotate_last.onnx'
        self.classes = ['barcode', 'qr', 'dm']
        cuda = False
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        self.session = onnxruntime.InferenceSession(self.weight, providers=providers)

    def predict(self, image) -> list:
        shapes = []
        height = int(image.shape[0] / 32 + 0.5) * 32
        width = int(image.shape[1] / 32 + 0.5) * 32
        image, ratio, (dw, dh) = letterbox(image, (height, width), (0, 0, 0), False)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[..., None]
        image = image.transpose((2, 0, 1))
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
            box[4] = (box[4] -dw) / ratio[0]
            box[5] = (box[5] -dh) / ratio[0]
            box[6] = (box[6] -dw) / ratio[0]
            box[7] = (box[7] -dh) / ratio[0]
            shapes.append(self.get_shape(self.classes[cl] if self.classes else str(cl), 
                            [[box[0], box[1]], [box[2], box[3]], [box[4], box[5]], [box[6], box[7]]], 'polygon'))
        return shapes
