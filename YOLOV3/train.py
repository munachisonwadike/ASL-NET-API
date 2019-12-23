import argparse
import test 
import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from models import *
from utils.datasets import *
from utils.utils import *
from visdom import Visdom

weights_dir = 'asl_weights' + os.sep #weights directory
best = weights_dir + 'best.pt' # checkpoint of weights with best mAP
last = weights_dir + 'last.pt' # checkpoint of weights from most recent forward pass 
results_fl = 'results.txt'

# Hyperparameters 
hyper_param = { 'cls': 27.76,  # cls loss gain  (CE=~1.0, uCE=~20)
        'cls_pw': 1.446,  # cls BCELoss positive_weight
        'degrees': 1.113,  # image rotation (+/- deg)
        'fl_gamma': 0.5,  # focal loss gamma
        'giou': 1.582,  # giou loss gain
        'hsv_h': 0.01,  # image HSV-Hue augmentation (fraction)
        'hsv_s': 0.5703,  # image HSV-Saturation augmentation (fraction)
        'hsv_v': 0.3174,  # image HSV-Value augmentation (fraction)
        'iou_t': 0.2635,  # iou training threshold
        'lr0': 0.002324,  # initial learning rate (SGD=1E-3, Adam=9E-5)
        'lrf': -4.,  # final LambdaLR learning rate = lr0 * (10 ** lrf)
        'momentum': 0.97,  # SGD momentum
        'obj': 21.35,  # obj loss gain (*=80 for uBCE with 80 classes)
        'obj_pw': 3.941,  # obj BCELoss positive_weight
        'scale': 0.1059,  # image scale (+/- gain)
        'shear': 0.5768,  # image shear (+/- deg)
        'translate': 0.06797,  # image translation (+/- fraction)
        'weight_decay': 0.0004569 }  # optimizer weight decay
       
# Visdom class to visualise training loss
class VisdomLinePlotter(object):
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, title_name, x, y):
        y=y.item()
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')
plotter = VisdomLinePlotter(env_name='Plots')


# Prebias function to train output bias layers for 1 epoch and create new backbone
def run_prebias():
    if args.prebias:
        train()  # results saved to last.pt after 1 epoch
        backbone_generate(last)  # backbone is saved as backbone.pt (see utils/utils)
        args.weights = weights_dir + 'backbone.pt' # set train to continue from backbone.pt
        args.prebias = False  


# Main training function
def train():
    # Remove any prior bounding box predictions generated by this code
    for file in glob.glob('*_batch*.jpg') + glob.glob(results_fl):
        os.remove(file)

    acc = args.accumulate  
    batch_size = args.batch_size
    cfg_file = args.cfg
    data = args.data
    if args.prebias:
        epochs = 1
    else:
        epochs = args.epochs 
       
    image_size = args.img_size
    weights_ = args.weights  

    if 'pw' not in args.arc:  # remove BCELoss positive weights
        hyper_param['cls_pw'] = 1.
        hyper_param['obj_pw'] = 1.
    
    seed_init()
    
    # For multiscale training - set the maximum image size and min image size, and set the 
    # starting image size to the maximum
    min_image_size = round(image_size / 32 / 1.5) + 1
    max_image_size = round(image_size / 32 * 1.5) - 1
    image_size = max_image_size * 32  

    # Inform
    print('Using multi-scale training %g - %g' % (min_image_size * 32, image_size))
 

    data_parsed = data_cfg_parser(data)
    train_data = data_parsed['train']
    num_classes = int(data_parsed['classes'])  

    #Inform 
    print('Training  data located at %s' % (train_data))


    # Initialize model
    model = YOLOV3(cfg_file, arc=args.arc).to(device)

    # Optimizer for learning schedule
    parameter_group0, parameter_group1 = [], []  
    for key, value in dict(model.named_parameters()).items():
        if 'Conv2d.weight' in key:
            parameter_group1 += [value]  # parameter group 1 (apply weight_decay)
        else:
            parameter_group0 += [value]  # parameter group 0

    if args.adam:
        optimizer = optim.Adam(parameter_group0, lr=hyper_param['lr0'])
    else:
        optimizer = optim.SGD(parameter_group0, lr=hyper_param['lr0'], momentum=hyper_param['momentum'], nesterov=True)
    # add parameter_group1 with weight_decay
    optimizer.add_param_group({'params': parameter_group1, 'weight_decay': hyper_param['weight_decay']}) 
    del parameter_group0, parameter_group1


    # backbone reaches to cutoff layer
    cutoff = -1  
    epoch_start = 0
    best_fit = float('inf')

    try_download(weights_) # will only run if not os.path.isfile(weights)
    
    if weights_.endswith('.pt'):  # pytorch format
        # e.g 'last.pt', 'yolov3-spp.pt'

        chkpt = torch.load(weights_, map_location=device)
        
       
        # Model state dict loading
        chkpt['model'] = {key: value for key, value in chkpt['model'].items() if model.state_dict()[key].numel() == value.numel()}
        model.load_state_dict(chkpt['model'], strict=False)

        # optimizer state dict loading
        if chkpt['optimizer'] is not None:
            optimizer.load_state_dict(chkpt['optimizer'])
            best_fit = chkpt['best_fitness']

        # load results  and write to results.txt
        if chkpt.get('training_results') is not None:
            with open(results_fl, 'w') as fl:
                fl.write(chkpt['training_results']) 

        epoch_start = chkpt['epoch'] + 1
        del chkpt

    
    elif len(weights_) > 0:  # darknet format e.g  'yolov3.weights', 'yolov3-tiny.conv.15'
        cutoff = load_darknet_weights(model, weights_)

    # transfer learning, in the case of prebias set to true, for one epoch
    if args.transfer or args.prebias: 
        # yolo layer size (i.e. 255)
        num_filters = int(model.module_defs[model.yolo_layers[0] - 1]['filters']) 
        if args.prebias:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 100  # lr gain
                if param_group.get('momentum') is not None: 
                    param_group['momentum'] *= 0.9

        for param in model.parameters():
            if args.prebias and param.numel() == num_filters:  
                param.requires_grad = True
            elif args.transfer and p.shape[0] == num_filters:   
                param.requires_grad = True
            else:  
                param.requires_grad = False
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[round(args.epochs * x) for x in [0.8, 0.9]], gamma=0.1)
    scheduler.last_epoch = epoch_start - 1

    # Dataset
    dataset = ImagesPlusLabelLoader(train_data,
                                  image_size,
                                  batch_size,
                                  augment=True,
                                  hyp=hyper_param,  
                                  rect=args.rect,   
                                  image_weights=args.img_weights,
                                  cache_labels=False if epochs > 10 else False,
                                  cache_images=False if args.prebias else args.cache_images)
    #  The length of the loader will adapt to the batch_size. So if your train dataset has 1000 samples
    # and you use a batch_size of 10, the loader will have the length 100.
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=min([os.cpu_count(), batch_size, 16]),
                                             shuffle=not args.rect, 
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)


   
    model.nc = num_classes   
    model.arc = args.arc  
    model.hyp = hyper_param   

    torch_utils.model_info(model, report='summary')  
    
    len_dataloader = len(dataloader)

    #Set the mAP values per class to zero to be changed as training progresses
    maps = np.zeros(num_classes)  # mAP per class
    # Similarly set a tuple that will store testing results
    # Order- 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    results_tuple = (0, 0, 0, 0, 0, 0, 0) 
    
    
    start_time = time.time()
    print('Starting %s for %g epochs...' % ('prebias' if args.prebias else 'training', epochs))
    
    epoch = 0

    for epoch in range(epoch_start, epochs):  
        model.train()
        print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'image_size'))

       
        # Update image weights 
        if dataset.image_weights:
            class_weights = model.class_weights.cpu().numpy() * (1 - maps) ** 2  # class weights
            
            image_weights = labels_to_image_weights(dataset.labels, nc=num_classes, class_weights=class_weights)
            
            dataset.indices = random.choices(range(dataset.num_imgs), weights=image_weights, k=dataset.num_imgs)  # rand weighted idx
        
        
        mean_loss = torch.zeros(4).to(device)   

        # import progress bar and start run through the batches
        pbar = tqdm(enumerate(dataloader), total=len_dataloader) 
        
        for index, (images, targets, paths, _) in pbar:  


            batch_count = index + len_dataloader * epoch  
            
            images = images.to(device)
            
            targets = targets.to(device)

            # Multi-Scale training -- adjust (67% - 150%) every 10 batches
            if batch_count / acc % 10 == 0: 
                image_size = random.randrange(min_image_size, max_image_size + 1) * 32
            
            scale_factor = image_size / max(images.shape[2:])   
            if scale_factor != 1:
                new_shape = [math.ceil(x * scale_factor / 32.) * 32 for x in images.shape[2:]]  # stretched to 32-multiple
                images = F.interpolate(images, size=new_shape, mode='bilinear', align_corners=False)

            # Plot images with bounding boxes
            if batch_count == 0:
                fname = 'train_batch%g.jpg' % index
                image_plot(imgs=images, targets=targets, paths=paths, filename=fname)
            


            prediction = model(images)


            loss, loss_items = compute_loss(prediction, targets, model)

            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss_items)
                return results_tuple

            # Scale loss by nominal batch_size
            loss *= batch_size / 64

            loss.backward()
        
            if batch_count % acc == 0:
                optimizer.step()
                optimizer.zero_grad()
            

            mean_loss = (mean_loss * index + loss_items) / (index + 1)   
            mem_avail = torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0   
            
            # set the values for the description to print to terminal 
            description = ('%10s' * 2 + '%10.3g' * 6) % (
                '%g/%g' % (epoch, epochs - 1), '%.3gG' % mem_avail, *mean_loss, len(targets), image_size)
            pbar.set_description(description)

        plotter.plot("loss", "train", "Train YOLOV3", epoch, loss)

        scheduler.step()

        # show the results from each epoch after processing them
        final_epoch = epoch + 1 == epochs
        if args.prebias:
            print_model_biases(model)
        else:
            # Calculate mAP 
            if not epoch > 200:
                # Test final epoch - can skip first 10 if args.nosave
                if not (args.notest or (args.nosave and epoch < 10)) or final_epoch:
                    with torch.no_grad():
                        results_tuple, maps = test.test(cfg_file,
                                                  data,
                                                  batch_size=batch_size,
                                                  img_size=args.img_size,
                                                  model=model,
                                                  conf_thres=0.001 if final_epoch and epoch > 0 else 0.1,  
                                                  save_json=final_epoch and epoch > 0 and 'coco.data' in data)
                    

        with open(results_fl, 'a') as file:
            file.write(description + '%10.3g' * 7 % results_tuple + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)
        
        fit = sum(results_tuple[4:])

        if fit < best_fit:
            best_fit = fit

        save = (not args.nosave) or (final_epoch and not args.evolve) or args.prebias
        if save:
            with open(results_fl, 'r') as f:
                chkpt = {'epoch': epoch,
                         'best_fitness': best_fit,
                         'training_results': f.read(),
                         'model': model.module.state_dict() if type(
                             model) is nn.parallel.DistributedDataParallel else model.state_dict(),
                         'optimizer': None if final_epoch else optimizer.state_dict()}

            torch.save(chkpt, last)

            if best_fit == fit:
                torch.save(chkpt, best)

            # backup every ten epochs
            if epoch > 0 and epoch % 10 == 0:
                torch.save(chkpt, weights_dir + 'backup%g.pt' % epoch)
            del chkpt
 
    if len(args.name):
        os.rename('results.txt', 'results_%s.txt' % args.name)
        os.rename(weights_dir + 'best.pt', weights_dir + 'best_%s.pt' % opt.name)

    results_plotter()  # save as results.png
    print('%g epochs completed in %.3f hours.\n' % (epoch - epoch_start + 1, (time.time() - start_time) / 3600))
    dist.destroy_process_group() if torch.cuda.device_count() > 1 else None
    torch.cuda.empty_cache()
    
    return results_tuple




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--accumulate', type=int, default=2, help='batches to accumulate before optimizing')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    parser.add_argument('--arc', type=str, default='defaultpw', help='yolo architecture')  # # default with positive weights
    parser.add_argument('--batch-size', type=int, default=6)  # effective bs = batch_size * accumulate = 8 *  = 16
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/asl_images/asl.data', help='*.data file path')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--epochs', type=int, default=273)   
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--img-weights', action='store_true', help='select training images by weight')
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--prebias', action='store_true', help='transfer-learn yolo biases prior to training')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', action='store_true', help='resume training from last.pt')
    parser.add_argument('--transfer', action='store_true', help='transfer learning')
    parser.add_argument('--var', type=float, help='debug variable')
    parser.add_argument('--weights', type=str, default='', help='initial weights')  # i.e. weights/darknet.53.conv.74

    args = parser.parse_args()
    args.weights = last if args.resume else args.weights
    print(args)
    device = torch_utils.select_device(args.device)

    print
    args.evolve = False
    run_prebias()   
    train()   