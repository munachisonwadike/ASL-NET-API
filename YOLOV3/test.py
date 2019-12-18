import argparse
import json


from models import *
from torch.utils.data import DataLoader
from utils.datasets import *
from utils.utils import *


def test(cfg,
         data,
         weights=None,
         batch_size=16,
         img_size=416,
         iou_thres=0.5,
         conf_thres=0.001,
         nms_thres=0.5,
         save_json=False,
         model=None):
    # Initialize/load model and set device
    if model is None:
        device = torch_utils.select_device(opt.device)
        verbose = True

        # Initialize model
        model = YOLOV3(cfg, img_size).to(device)

        # Load weights
        try_download(weights)
        if weights.endswith('.pt'):  # pytorch format
            model.load_state_dict(torch.load(weights, map_location=device)['model'])
        else:  # darknet format
            _ = load_darknet_weights(model, weights)

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    else:
        device = next(model.parameters()).device  # get model device
        verbose = False

    # Configure run
    data = data_cfg_parser(data)
    num_classes = int(data['classes'])   
    test_path = data['valid']  
    class_names = classes_load(data['names'])   

    # Dataloader
    dataset = ImagesPlusLabelLoader(test_path, img_size, batch_size)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=min([os.cpu_count(), batch_size, 16]),
                            pin_memory=True,
                            collate_fn=dataset.collate_fn)

    
    model.eval()
    coco91class = coco80_to_coco91_class()
    out_string = ('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP', 'F1')
    p, r, f1, mean_p, mean_r, mean_AP, mean_f1 = 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3)
    jdict, stats, ap, ap_class = [], [], [], []

    seen = 0
    for batch_index, (images, targs, paths, shapes) in enumerate(tqdm(dataloader, desc=out_string)):
        
        targs = targs.to(device)
        images = images.to(device)
        _, _, height, width = images.shape  # batch size, channels, height, width

        if batch_index == 0 and not os.path.exists('test_batch0.jpg'):
            image_plot(imgs=images, targets=targs, paths=paths, filename='test_batch0.jpg')

        inf_out, train_out = model(images)   

        if hasattr(model, 'hyp'):  # if model has loss hyperparameters
            loss += compute_loss(train_out, targs, model)[1][:3].cpu()  # GIoU, obj, cls

        output = non_max_suppression(inf_out, conf_thres=conf_thres, nms_thres=nms_thres)

        for stats_per_image, pred in enumerate(output):
            labels = targs[targs[:, 0] == stats_per_image, 1:]
            num_labels = len(labels)
            target_class = labels[:, 0].tolist() if num_labels else []  
            seen += 1

            if pred is None:
                if num_labels:
                    stats.append(([], torch.Tensor(), torch.Tensor(), target_class))
                continue

            # Clip boxes to image bounds
            clip_coords(pred, (height, width))

            # Assign all predictions as incorrect
            correct = [0] * len(pred)
            if num_labels:
                detected = []
                target_class_tensor = labels[:, 0]

                target_box = xywh2xyxy(labels[:, 1:5])
                target_box[:, [0, 2]] *= width
                target_box[:, [1, 3]] *= height

                # Search for correct predictions
                for i, (*pred_box, pred_conf, pred_class_conf, pred_class) in enumerate(pred):

                    if len(detected) == num_labels:
                        break
 
                    if pred_class.item() not in target_class:
                        continue

                    # Best iou, index between pred and targets
                    ind = (pred_class == target_class_tensor).nonzero().view(-1)
                    iou, max_ = bbox_iou(pred_box, target_box[ind]).max(0)

                    # If iou > threshold and class is correct mark as correct
                    if iou > iou_thres and ind[max_] not in detected:  # and pred_class == target_class[max_]:
                        correct[i] = 1
                        detected.append(ind[max_])

            # Append statistics (correct, conf, pred_class, target_class)
            stats.append((correct, pred[:, 4].cpu(), pred[:, 6].cpu(), target_class))

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in list(zip(*stats))]  # to numpy
    if len(stats):
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        mean_p, mean_r, mean_AP, mean_f1 = p.mean(), r.mean(), ap.mean(), f1.mean()
        num_targets = np.bincount(stats[3].astype(np.int64), minlength=num_classes)  # number of targets per class
    else:
        num_targets = torch.zeros(1)

    print_format = '%20s' + '%10.3g' * 6 
    print( print_format  % ('all', seen, num_targets.sum(), mean_p, mean_r, mean_AP, mean_f1))

    # Class-wise results
    if verbose and num_classes > 1 and len(stats):
        for index, class_ in enumerate(ap_class):
            print( print_format  % (class_names[class_], seen, num_targets[class_], p[index], r[index], ap[index], f1[index]))
 

    # Return results
    maps = np.zeros(num_classes) + mean_AP
    for index, class_ in enumerate(ap_class):
        maps[class_] = ap[index]
    return (mean_p, mean_r, mean_AP, mean_f1, *(loss / len(dataloader)).tolist()), maps


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/asl_images/asl.data', help='coco.data file path')
    parser.add_argument('--weights', type=str, default='asl_weights/backup70.pt', help='path to weights file')
    parser.add_argument('--batch-size', type=int, default=16, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='iou threshold required to qualify as detected')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        test(opt.cfg,
             opt.data,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.iou_thres,
             opt.conf_thres,
             opt.nms_thres,
             opt.save_json)
