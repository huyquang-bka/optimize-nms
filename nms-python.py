import time
import numpy as np

def numpy_unsqueeze(x):
    return np.expand_dims(x, axis=1)

def numpy_chunk(x, n, dim):
    return np.split(x, n, dim)

def xywh2xyxy(x: np.ndarray):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.copy()
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def nms(boxes, scores, iou_threshold):
    # Sort boxes by scores in descending order
    order = scores.argsort()[::-1]
    boxes = boxes[order]
    scores = scores[order]

    # Initialize a list to store selected indices after NMS
    selected_indices = []

    while len(boxes) > 0:
        # Pick the box with the highest score
        selected_indices.append(order[0])
        
        # Calculate IoU between the picked box and the remaining boxes
        iou = calculate_iou(boxes[0], boxes[1:])
        
        # Filter out boxes with high IoU
        mask = iou <= iou_threshold
        boxes = boxes[1:][mask]
        scores = scores[1:][mask]
        order = order[1:][mask]

    return selected_indices

def calculate_iou(box, boxes):
    # Calculate intersection and union areas
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - intersection
    
    # Calculate IoU
    iou = intersection / (union + 1e-6)  # Add a small value to avoid division by zero
    
    return iou


def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    # (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(
    #     2, 2), box2.unsqueeze(0).chunk(2, 2)
    # inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = numpy_chunk(numpy_unsqueeze(box1), 2, 2), numpy_chunk(numpy_unsqueeze(box2), 2, 2)
    inter = (np.minimum(a2, b2) - np.maximum(a1, b1)).clip(0).prod(2)
    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        labels=(),
        max_det=300,
        nm=0,  # number of masks
):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    # YOLOv5 model in validation model, output = (inference_out, loss_out)
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]  # select only inference output

    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()

    mi = 5 + nc  # mask start index
    output = [np.zeros((0, 6 + nm))] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = np.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = np.concatenate((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        # center_x, center_y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])
        mask = x[:, mi:]  # zero columns if no masks

        # conf, j = x[:, 5:mi].max(1, keepdim=True)
        conf = np.max(x[:, 5:mi], axis=1, keepdims=True)
        j = np.argmax(x[:, 5:mi], axis=1, keepdims=True)
        x = np.concatenate((box, conf, j, mask), 1)[
            conf.flatten() > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == np.array(classes)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        # sort by confidence and remove excess boxes
        x = x[x[:, 4].argsort()[::-1][:max_nms]]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        # boxes (offset by class), scores
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections
        output[xi] = x[i]
    return output

if __name__ == "__main__":
    pred = np.load("pred.npy")
    t = time.time()
    pred = non_max_suppression(pred, 0.25, 0.45)
    print(time.time() - t)
    print(pred)