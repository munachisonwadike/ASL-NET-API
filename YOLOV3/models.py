import torch.nn.functional as F

from utils.google_utils import *
from utils.parse_config import *
from utils.utils import *


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, inpt):
        return inpt * torch.sigmoid(inpt)


class YOLOLayer(nn.Module):
    def __init__(self, anchors, num_classes, img_size, yolo_index, arc):
        super(YOLOLayer, self).__init__()

        self.anchors = torch.Tensor(anchors)
        self.num_anchors = len(anchors)   
        self.num_classes = num_classes  
        self.num_x = 0  # number of x gridpoints
        self.num_y = 0  # ""     "" y gridpoints
        self.arc = arc

    def forward(self, prediction, img_size, var=None):
        
        bs, num_y, num_x = prediction.shape[0], prediction.shape[-2], prediction.shape[-1]
        if (self.num_x, self.num_y) != (num_x, num_y):
            create_grids(self, img_size, (num_x, num_y), prediction.device, prediction.dtype)

        ### stopped here - may need to edit p and currently testing with inference (detect.py- run with pretrained weights since I don't have time to fix my weights)
        prediction = prediction.view(bs, self.num_anchors, self.num_classes + 5, self.num_y, self.num_x).permute(0, 1, 3, 4, 2).contiguous()  

        if self.training:
            return prediction
        else:  # inference
            inference_output = prediction.clone()  # inference output
            inference_output[..., 0:2] = torch.sigmoid(inference_output[..., 0:2]) + self.grid_xy  # xy
            inference_output[..., 2:4] = torch.exp(inference_output[..., 2:4]) * self.anchor_wh  # wh yolo method
            inference_output[..., :4] *= self.stride

            if 'default' in self.arc:  # seperate obj and cls
                torch.sigmoid_(inference_output[..., 4:])

            if self.num_classes == 1: # single-class model 
                inference_output[..., 5] = 1  

            # reshape from [1, 3, 13, 13, 85] to [1, 507, 85]
            return inference_output.view(bs, -1, 5 + self.num_classes), prediction



# YOLOv3 object detection model
class YOLOV3(nn.Module):

    def __init__(self, cfg, img_size=(416, 416), arc='default'):
        super(YOLOV3, self).__init__()

        self.module_defs = model_cfg_parser(cfg)
        self.module_list, self.routs = generate_modules(self.module_defs, img_size, arc)
        self.yolo_layers = get_yolo_layers(self)
 
        self.version = np.array([0, 2, 5], dtype=np.int32)  # (int32) version info: major, minor, revision
        self.seen = np.array([0], dtype=np.int64)  # (int64) number of images seen during training

    def forward(self, inpt, var=None):
        img_size = inpt.shape[-2:]
        layer_outputs = []
        output = []

        for index, (module_definition, module) in enumerate(zip(self.module_defs, self.module_list)):

            model_type = module_definition['type']
            if model_type in ['convolutional', 'upsample', 'maxpool']:
                inpt = module(inpt)
            elif model_type == 'route':
                layers = [int(inpt) for inpt in module_definition['layers'].split(',')]

                if len(layers) == 1:
                    inpt = layer_outputs[layers[0]]
                else:
                    try:
                        inpt = torch.cat([layer_outputs[index] for index in layers], 1)
                    except:   
                        layer_outputs[layers[1]] = F.interpolate(layer_outputs[layers[1]], scale_factor=[0.5, 0.5])
                        inpt = torch.cat([layer_outputs[index] for index in layers], 1)

            elif model_type == 'shortcut':
                inpt = inpt + layer_outputs[int(module_definition['from'])]
            elif model_type == 'yolo':
                inpt = module(inpt, img_size)
                output.append(inpt)
            layer_outputs.append(inpt if index in self.routs else [])

        if self.training:
            return output
        else:
            inference_output, prediction = list(zip(*output))  # inference output, training output
            return torch.cat(inference_output, 1), prediction


# .pt to .weights and vice versa
def convert(cfg='cfg/yolov3-spp.cfg', weights='weights/yolov3-spp.weights'):
    # Initialize model
    model = Darknet(cfg)

    #Converts between PyTorch to Darknet format
    if weights.endswith('.pt'):   

        model.load_state_dict(torch.load(weights, map_location='cpu')['model'])
        save_weights(model, path='converted.weights', cutoff=-1)
        print("Success: converted '%s' to 'converted.weights'" % weights)

    #Converts between Darknet to PyTorch format
    elif weights.endswith('.weights'):   
        load_darknet_weights(model, weights)

        chkpt = {'epoch': -1,
                 'best_fitness': None,
                 'training_results': None,
                 'model': model.state_dict(),
                 'optimizer': None}

        torch.save(chkpt, 'converted.pt')

        print("Success: converted '%s' to 'converted.pt'" % weights)

    # None of the supported formats
    else:
        print('Error: extension not supported.')


def create_grids(self, img_size=416, ng=(13, 13), device='cpu', type=torch.float32):
    num_x, num_y = ng  # x and y grid size as defined in yolo_layer class
    self.img_size = max(img_size)
    self.stride = self.img_size / max(ng)

     
    y_offset, x_offset = torch.meshgrid([torch.arange(num_y), torch.arange(num_x)])
    self.grid_xy = torch.stack((x_offset, y_offset), 2).to(device).type(type).view((1, 1, num_y, num_x, 2)) #grid_xy initialised in yolo_layer class
 
    self.anchor_vec = self.anchors.to(device) / self.stride
    self.anchor_wh = self.anchor_vec.view(1, self.num_anchors, 1, 1, 2).to(device).type(type)
    self.ng = torch.Tensor(ng).to(device)
    self.num_x = num_x
    self.num_y = num_y


# Create the modules needed from the model cfg file and return as a list of blocks
def generate_modules(module_defs, img_size, arc):

    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams['channels'])]
    module_list = nn.ModuleList()
    routs = []  # list of layers which rout to deeper layes
    yolo_index = -1

    for index, module_definition in enumerate(module_defs):
        modules = nn.Sequential()

        # generate convolutional layer model
        if module_definition['type'] == 'convolutional':
            bn = int(module_definition['batch_normalize'])
            filters = int(module_definition['filters'])
            kernel_size = int(module_definition['size'])
            pad = (kernel_size - 1) // 2 if int(module_definition['pad']) else 0
            modules.add_module('Conv2d', nn.Conv2d(in_channels=output_filters[-1],
                                                   out_channels=filters,
                                                   kernel_size=kernel_size,
                                                   stride=int(module_definition['stride']),
                                                   padding=pad,
                                                   bias=not bn))
            if bn:
                modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.1))
            if module_definition['activation'] == 'leaky':  
                modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))

        # generate maxpool layer model
        elif module_definition['type'] == 'maxpool':
            kernel_size = int(module_definition['size'])
            stride = int(module_definition['stride'])
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            if kernel_size == 2 and stride == 1:  # yolov3-tiny
                modules.add_module('ZeroPad2d', nn.ZeroPad2d((0, 1, 0, 1)))
                modules.add_module('MaxPool2d', maxpool)
            else:
                modules = maxpool

        # generate upsample layer model
        elif module_definition['type'] == 'upsample':
            modules = nn.Upsample(scale_factor=int(module_definition['stride']), mode='nearest')

        # generate route layer model
        elif module_definition['type'] == 'route':  # nn.Sequential() placeholder for 'route' layer
            layers = [int(x) for x in module_definition['layers'].split(',')]
            filters = sum([output_filters[i + 1 if i > 0 else i] for i in layers])
            routs.extend([l if l > 0 else l + index for l in layers])

        # generate shortcut layer model
        elif module_definition['type'] == 'shortcut':  
            filters = output_filters[int(module_definition['from'])]
            layer = int(module_definition['from'])
            routs.extend([index + layer if layer < 0 else layer])

        # generate reorg layer model
        elif module_definition['type'] == 'reorg3d':  # yolov3-spp-pan-scale
            pass

        # generate yolo layer model
        elif module_definition['type'] == 'yolo':
            yolo_index += 1
            mask = [int(x) for x in module_definition['mask'].split(',')] 
            modules = YOLOLayer(anchors=module_definition['anchors'][mask],  
                                num_classes=int(module_definition['classes']),  
                                img_size=img_size,   
                                yolo_index=yolo_index, 
                                arc=arc) 

            try:
                if arc == 'defaultpw' or arc == 'Fdefaultpw':  # default with positive weights
                    b = [-4, -3.6]  # obj, cls
                elif arc == 'default':  # default no pw (40 cls, 80 obj)
                    b = [-5.5, -4.0]
                elif arc == 'uBCE':  # unified BCE (80 classes)
                    b = [0, -8.5]
                elif arc == 'uCE':  # unified CE (1 background + 80 classes)
                    b = [10, -0.1]
                elif arc == 'Fdefault':  # Focal default no pw (28 cls, 21 obj, no pw)
                    b = [-2.1, -1.8]
                elif arc == 'uFBCE' or arc == 'uFBCEpw':  # unified FocalBCE (5120 obj, 80 classes)
                    b = [0, -6.5]
                elif arc == 'uFCE':  # unified FocalCE (64 cls, 1 background + 80 classes)
                    b = [7.7, -1.1]


                bias = module_list[-1][0].bias.view(len(mask), -1)  # 255 to 3x85
                bias[:, 4] += b[0] - bias[:, 4].mean()  # obj
                bias[:, 5:] += b[1] - bias[:, 5:].mean()  # cls
                module_list[-1][0].bias = torch.nn.Parameter(bias.view(-1))
            except:
                print('WARNING: smart bias initialization failed.')

        else:
            print('WARNING: Unrecognized Layer Type: ' + module_definition['type'])

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return module_list, routs


def get_yolo_layers(model):

    return [index for index, mod_def in enumerate(model.module_defs) if mod_def['type'] == 'yolo']  # [82, 94, 106] for yolov3

# Parse and loads .weights files, darknet format
def load_darknet_weights(self, weights, cutoff=-1):

    # Establish cutoffs (load layers between 0 and cutoff. if cutoff = -1 all are loaded)
    fl = Path(weights).name
    if fl == 'darknet53.conv.74':
        cutoff = 75
    elif fl == 'yolov3-tiny.conv.15':
        cutoff = 15

    # Read weights file
    with open(weights, 'rb') as file:
        self.version = np.fromfile(file, dtype=np.int32, count=3)   
        self.seen = np.fromfile(file, dtype=np.int64, count=1)   

        weights = np.fromfile(file, dtype=np.float32)   

    ptr = 0
    for i, (module_definition, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
        if module_definition['type'] == 'convolutional':
            conv_layer = module[0]
            if module_definition['batch_normalize']:
                # Load BN bias, weights, running mean and running variance
                bn_layer = module[1]
                num_b = bn_layer.bias.numel()  # Number of biases
                # Bias
                bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias)
                bn_layer.bias.data.copy_(bn_b)
                ptr += num_b
                # Weight
                bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight)
                bn_layer.weight.data.copy_(bn_w)
                ptr += num_b
                # Running Mean
                bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                bn_layer.running_mean.data.copy_(bn_rm)
                ptr += num_b
                # Running Var
                bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                bn_layer.running_var.data.copy_(bn_rv)
                ptr += num_b
            else:
                # Load conv. bias
                num_b = conv_layer.bias.numel()
                conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias)
                conv_layer.bias.data.copy_(conv_b)
                ptr += num_b
                
            # Load conv. weights
            num_weights = conv_layer.weight.numel()
            conv_w = torch.from_numpy(weights[ptr:ptr + num_weights]).view_as(conv_layer.weight)
            conv_layer.weight.data.copy_(conv_w)
            ptr += num_weights

    return cutoff

# Download weights if not available locally
def try_download(weights):

    msg = weights + ' missing, download from https://drive.google.com/drive/folders/1uxgUBemJVw9wZsdpboYbzUN4bcRhsuAI'
    if weights and not os.path.isfile(weights):
        file = Path(weights).name

        if file == 'yolov3-spp.weights':
            gdrive_download(id='1oPCHKsM2JpM-zgyepQciGli9X0MTsJCO', name=weights)
        elif file == 'yolov3-spp.pt':
            gdrive_download(id='1vFlbJ_dXPvtwaLLOu-twnjK4exdFiQ73', name=weights)
        elif file == 'yolov3.pt':
            gdrive_download(id='11uy0ybbOXA2hc-NJkJbbbkDwNX1QZDlz', name=weights)
        elif file == 'yolov3-tiny.pt':
            gdrive_download(id='1qKSgejNeNczgNNiCn9ZF_o55GFk1DjY_', name=weights)
        elif file == 'darknet53.conv.74':
            gdrive_download(id='18xqvs_uwAqfTXp-LJCYLYNHBOcrwbrp0', name=weights)
        elif file == 'yolov3-tiny.conv.15':
            gdrive_download(id='140PnSedCsGGgu3rOD6Ez4oI6cdDzerLC', name=weights)

        else:
            try:   
                url = 'https://pjreddie.com/media/files/' + file
                print('Downloading ' + url)
                os.system('curl -f ' + url + ' -o ' + weights)
            except IOError:
                print(msg)
                os.system('rm ' + weights)  

        assert os.path.exists(weights), msg  