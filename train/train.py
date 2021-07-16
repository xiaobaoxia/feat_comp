# The implementation of GDN is inherited from
# https://github.com/jorge-pessoa/pytorch-gdn,
# under the MIT License.
#
# This file is being made available under the BSD License.
# Copyright (c) 2021 Yueyu Hu
from networks import *

result_saved_path = 'result'
# batch_size = 2
SCALE = (256,256)

class Logger(object):
    '''Save training process to log file with simple plot function.'''
    def __init__(self, fpath, title=None, resume=False):
        self.file = None
        self.resume = resume
        self.title = '' if title == None else title
        if fpath is not None:
            if resume:
                self.file = open(fpath, 'r')
                name = self.file.readline()
                self.names = name.rstrip().split('\t')
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []

                for numbers in self.file:
                    numbers = numbers.rstrip().split('\t')
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(numbers[i])
                self.file.close()
                self.file = open(fpath, 'a')
            else:
                self.file = open(fpath, 'w')

    def set_names(self, names):
        if self.resume:
            pass
        # initialize numbers as empty list
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write('\t')
            self.numbers[name] = []
        self.file.write('\n')
        self.file.flush()


    def append(self, numbers):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for index, num in enumerate(numbers):
            self.file.write("{0:.6f}".format(num))
            self.file.write('\t')
            self.numbers[self.names[index]].append(num)
        self.file.write('\n')
        self.file.flush()

    def plot(self, names=None):
        names = self.names if names == None else names
        numbers = self.numbers
        for _, name in enumerate(names):
            x = np.arange(len(numbers[name]))
            plt.plot(x, np.asarray(numbers[name]))
        plt.legend([self.title + '(' + name + ')' for name in names])
        plt.grid(True)

    def close(self):
        if self.file is not None:
            self.file.close()


class Resize_target():
    def __init__(self,scale):
        self.resize = transforms.Resize(scale)

    def resize_bboxes(self,bboxes,scale_factor):
        bboxes = bboxes * scale_factor

        return bboxes

    def resize_masks(self,masks):
        masks = self.resize(masks)

        return masks


@torch.no_grad()
def test(net,epoch,stage,val_data,val_loader,val_transform):
    start_time = time.time()
    net.eval()
    list_test_eval_bpp = 0.
    list_test_v_mse = 0.
    list_test_v_psnr = 0.
    list_test_bpp1 = 0.
    list_test_bpp2 = 0.
    list_test_bpp3 = 0.
    cnt = 0

    resize_target = Resize_target(SCALE)
    # 可以用下面的方法生成新的标注
    coco = convert_to_coco_api(val_data, SCALE, resize_target, val_transform)
    iou_types = ["bbox","segm"]
    coco_evaluator = CocoEvaluator(coco, iou_types)

    with tqdm(total=len(val_data), desc=f'Epoch test {epoch + 1}/{500}', unit='img',ncols=100) as pbar:
        for i, (images, targets) in enumerate(val_loader):
            # todo：修改targets mask 大小与图片一致 需要缩放bbox和mask maskrcnn自带resize需要重写
            images = torch.stack([image.cuda() for image in images], dim=0)
            targets = [{k: v.cuda() if k != 'image_id' else v for k, v in t.items()} for t in targets]

            # todo：调整网络结构 组织targets，查看mask rcnn怎么训练
            eval_bpp,v_mse, v_psnr, x_hat, bpp1, bpp2, bpp3, seg_result = net(images,targets=None, mode='test', stage=4)

            outputs = [{k: v.cpu() for k, v in t.items()} for t in seg_result]

            res = {target["image_id"]: output for target, output in zip(targets, outputs)}

            coco_evaluator.update(res)

            eval_bpp = eval_bpp.mean()
            v_psnr = v_psnr.mean()
            v_mse = v_mse.mean()
            bpp1 = bpp1.mean()
            bpp2 = bpp2.mean()
            bpp3 = bpp3.mean()

            list_test_eval_bpp += eval_bpp.item()
            list_test_v_mse += v_mse.item()
            list_test_v_psnr += v_psnr.item()
            list_test_bpp1 += bpp1.item()
            list_test_bpp2 += bpp2.item()
            list_test_bpp3 += bpp3.item()

            cnt += 1

            # pbar.set_postfix(test_loss='{:.6f}'.format(test_loss.detach().cpu().numpy()))
            pbar.update(images.shape[0])

    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    eval_result = coco_evaluator.summarize()

    batch_val_bpp = list_test_eval_bpp / cnt
    batch_val_mse = list_test_v_mse / cnt
    batch_val_psnr = list_test_v_psnr / cnt
    batch_val_bpp1 = list_test_bpp1 / cnt
    batch_val_bpp2 = list_test_bpp2 / cnt
    batch_val_bpp3 = list_test_bpp3 / cnt
    timestamp = time.time()
    print('[Epoch %04d TEST %.1f seconds] eval_bpp: %.4e v_mse: %.4e v_psnr: %.4e bpp1: %.4e bpp2: %.4e bpp3: %.4e' % (
        epoch,
        timestamp - start_time,
        batch_val_bpp,
        batch_val_mse,
        batch_val_psnr,
        batch_val_bpp1,
        batch_val_bpp2,
        batch_val_bpp3
    ))
    net.train()

    return batch_val_bpp,batch_val_mse,batch_val_psnr,batch_val_bpp1,batch_val_bpp2,batch_val_bpp3,eval_result


name_dict = {
    'a_model.transform.1.weight': 'RE6_GPU0/analysis_transform_model/layre_1_mainA/kernel',
    'a_model.transform.1.bias': 'RE6_GPU0/analysis_transform_model/layre_1_mainA/bias',
    'a_model.transform.2.beta': 'RE6_GPU0/analysis_transform_model/layre_1_mainA/gdn/reparam_beta',
    'a_model.transform.2.gamma': 'RE6_GPU0/analysis_transform_model/layre_1_mainA/gdn/reparam_gamma',
    'a_model.transform.4.weight': 'RE6_GPU0/analysis_transform_model/layre_2_mainA/kernel',
    'a_model.transform.4.bias': 'RE6_GPU0/analysis_transform_model/layre_2_mainA/bias',
    'a_model.transform.5.beta': 'RE6_GPU0/analysis_transform_model/layre_2_mainA/gdn_1/reparam_beta',
    'a_model.transform.5.gamma': 'RE6_GPU0/analysis_transform_model/layre_2_mainA/gdn_1/reparam_gamma',
    'a_model.transform.7.weight': 'RE6_GPU0/analysis_transform_model/layre_3_mainA/kernel',
    'a_model.transform.7.bias': 'RE6_GPU0/analysis_transform_model/layre_3_mainA/bias',
    'a_model.transform.8.beta': 'RE6_GPU0/analysis_transform_model/layre_3_mainA/gdn_2/reparam_beta',
    'a_model.transform.8.gamma': 'RE6_GPU0/analysis_transform_model/layre_3_mainA/gdn_2/reparam_gamma',
    'a_model.transform.10.weight': 'RE6_GPU0/analysis_transform_model/layre_4_mainA/kernel',
    'a_model.transform.10.bias': 'RE6_GPU0/analysis_transform_model/layre_4_mainA/bias',

    's_model.transform.1.weight': 'RE6_GPU0/synthesis_transform_model/layer_1_mainS/kernel',
    's_model.transform.1.bias': 'RE6_GPU0/synthesis_transform_model/layer_1_mainS/bias',
    's_model.transform.2.beta': 'RE6_GPU0/synthesis_transform_model/layer_1_mainS/gdn_3/reparam_beta',
    's_model.transform.2.gamma': 'RE6_GPU0/synthesis_transform_model/layer_1_mainS/gdn_3/reparam_gamma',
    's_model.transform.4.weight': 'RE6_GPU0/synthesis_transform_model/layer_2_mainS/kernel',
    's_model.transform.4.bias': 'RE6_GPU0/synthesis_transform_model/layer_2_mainS/bias',
    's_model.transform.5.beta': 'RE6_GPU0/synthesis_transform_model/layer_2_mainS/gdn_4/reparam_beta',
    's_model.transform.5.gamma': 'RE6_GPU0/synthesis_transform_model/layer_2_mainS/gdn_4/reparam_gamma',
    's_model.transform.7.weight': 'RE6_GPU0/synthesis_transform_model/layer_3_mainS/kernel',
    's_model.transform.7.bias': 'RE6_GPU0/synthesis_transform_model/layer_3_mainS/bias',
    's_model.transform.8.beta': 'RE6_GPU0/synthesis_transform_model/layer_3_mainS/gdn_5/reparam_beta',
    's_model.transform.8.gamma': 'RE6_GPU0/synthesis_transform_model/layer_3_mainS/gdn_5/reparam_gamma',

    'ha_model_1.transform.0.weight': 'RE6_GPU0/h_analysis_transform_model_load/layer_1_h1a/kernel',
    'ha_model_1.transform.0.bias': 'RE6_GPU0/h_analysis_transform_model_load/layer_1_h1a/bias',
    'ha_model_1.transform.2.weight': 'RE6_GPU0/h_analysis_transform_model_load/layer_2_h1a/kernel',
    'ha_model_1.transform.2.bias': 'RE6_GPU0/h_analysis_transform_model_load/layer_2_h1a/bias',
    'ha_model_1.transform.4.weight': 'RE6_GPU0/h_analysis_transform_model_load/layer_3_h1a/kernel',
    'ha_model_1.transform.4.bias': 'RE6_GPU0/h_analysis_transform_model_load/layer_3_h1a/bias',
    'ha_model_1.transform.6.weight': 'RE6_GPU0/h_analysis_transform_model_load/layer_4_h1a/kernel',
    'ha_model_1.transform.6.bias': 'RE6_GPU0/h_analysis_transform_model_load/layer_4_h1a/bias',

    'ha_model_2.transform.0.weight': 'RE6_GPU0/h_analysis_transform_model_load_1/layer_1_h2a/kernel',
    'ha_model_2.transform.0.bias': 'RE6_GPU0/h_analysis_transform_model_load_1/layer_1_h2a/bias',
    'ha_model_2.transform.2.weight': 'RE6_GPU0/h_analysis_transform_model_load_1/layer_2_h2a/kernel',
    'ha_model_2.transform.2.bias': 'RE6_GPU0/h_analysis_transform_model_load_1/layer_2_h2a/bias',
    'ha_model_2.transform.4.weight': 'RE6_GPU0/h_analysis_transform_model_load_1/layer_3_h2a/kernel',
    'ha_model_2.transform.4.bias': 'RE6_GPU0/h_analysis_transform_model_load_1/layer_3_h2a/bias',
    'ha_model_2.transform.6.weight': 'RE6_GPU0/h_analysis_transform_model_load_1/layer_4_h2a/kernel',
    'ha_model_2.transform.6.bias': 'RE6_GPU0/h_analysis_transform_model_load_1/layer_4_h2a/bias',

    'hs_model_1.transform.0.weight': 'RE6_GPU0/h_synthesis_transform_model_load/layer_1_h1s/kernel',
    'hs_model_1.transform.0.bias': 'RE6_GPU0/h_synthesis_transform_model_load/layer_1_h1s/bias',
    'hs_model_1.transform.1.weight': 'RE6_GPU0/h_synthesis_transform_model_load/layer_2_h1s/kernel',
    'hs_model_1.transform.1.bias': 'RE6_GPU0/h_synthesis_transform_model_load/layer_2_h1s/bias',
    'hs_model_1.transform.3.weight': 'RE6_GPU0/h_synthesis_transform_model_load/layer_3_h1s/kernel',
    'hs_model_1.transform.3.bias': 'RE6_GPU0/h_synthesis_transform_model_load/layer_3_h1s/bias',
    'hs_model_1.transform.7.weight': 'RE6_GPU0/h_synthesis_transform_model_load/layer_4_h1s/kernel',
    'hs_model_1.transform.7.bias': 'RE6_GPU0/h_synthesis_transform_model_load/layer_4_h1s/bias',

    'hs_model_2.transform.0.weight': 'RE6_GPU0/h_synthesis_transform_model_load_1/layer_1_h2s/kernel',
    'hs_model_2.transform.0.bias': 'RE6_GPU0/h_synthesis_transform_model_load_1/layer_1_h2s/bias',
    'hs_model_2.transform.1.weight': 'RE6_GPU0/h_synthesis_transform_model_load_1/layer_2_h2s/kernel',
    'hs_model_2.transform.1.bias': 'RE6_GPU0/h_synthesis_transform_model_load_1/layer_2_h2s/bias',
    'hs_model_2.transform.3.weight': 'RE6_GPU0/h_synthesis_transform_model_load_1/layer_3_h2s/kernel',
    'hs_model_2.transform.3.bias': 'RE6_GPU0/h_synthesis_transform_model_load_1/layer_3_h2s/bias',
    'hs_model_2.transform.7.weight': 'RE6_GPU0/h_synthesis_transform_model_load_1/layer_4_h2s/kernel',
    'hs_model_2.transform.7.bias': 'RE6_GPU0/h_synthesis_transform_model_load_1/layer_4_h2s/bias',

    'prediction_model_2.transform.1.weight': 'RE6_GPU0/prediction_model_load/P_conv1_pred_2/kernel',
    'prediction_model_2.transform.1.bias': 'RE6_GPU0/prediction_model_load/P_conv1_pred_2/bias',
    'prediction_model_2.transform.4.weight': 'RE6_GPU0/prediction_model_load/P_conv2_pred_2/kernel',
    'prediction_model_2.transform.4.bias': 'RE6_GPU0/prediction_model_load/P_conv2_pred_2/bias',
    'prediction_model_2.transform.7.weight': 'RE6_GPU0/prediction_model_load/P_conv3_pred_2/kernel',
    'prediction_model_2.transform.7.bias': 'RE6_GPU0/prediction_model_load/P_conv3_pred_2/bias',
    'prediction_model_2.fc.weight': 'RE6_GPU0/prediction_model_load/P_fc_pred_2/kernel',
    'prediction_model_2.fc.bias': 'RE6_GPU0/prediction_model_load/P_fc_pred_2/bias',

    'prediction_model_3.transform.1.weight': 'RE6_GPU0/prediction_model_load_1/P_conv1_pred_3/kernel',
    'prediction_model_3.transform.1.bias': 'RE6_GPU0/prediction_model_load_1/P_conv1_pred_3/bias',
    'prediction_model_3.transform.4.weight': 'RE6_GPU0/prediction_model_load_1/P_conv2_pred_3/kernel',
    'prediction_model_3.transform.4.bias': 'RE6_GPU0/prediction_model_load_1/P_conv2_pred_3/bias',
    'prediction_model_3.transform.7.weight': 'RE6_GPU0/prediction_model_load_1/P_conv3_pred_3/kernel',
    'prediction_model_3.transform.7.bias': 'RE6_GPU0/prediction_model_load_1/P_conv3_pred_3/bias',
    'prediction_model_3.fc.weight': 'RE6_GPU0/prediction_model_load_1/P_fc_pred_3/kernel',
    'prediction_model_3.fc.bias': 'RE6_GPU0/prediction_model_load_1/P_fc_pred_3/bias',

    'recon_feat.layer_1.1.weight': 'RE6_GPU0/side_info_recon_model_load/layer_1_mainS_recon/kernel',
    'recon_feat.layer_1.1.bias': 'RE6_GPU0/side_info_recon_model_load/layer_1_mainS_recon/bias',
    'recon_feat.layer_1a.1.weight': 'RE6_GPU0/side_info_recon_model_load/layer_1a_mainS_recon/kernel',
    'recon_feat.layer_1a.1.bias': 'RE6_GPU0/side_info_recon_model_load/layer_1a_mainS_recon/bias',
    'recon_feat.layer_1b.1.weight': 'RE6_GPU0/side_info_recon_model_load/layer_1b_mainS_recon/kernel',
    'recon_feat.layer_1b.1.bias': 'RE6_GPU0/side_info_recon_model_load/layer_1b_mainS_recon/bias',
    'recon_image.layer_3_1.0.weight': 'RE6_GPU0/side_info_recon_model_load/layer_3_1_mainS_recon/kernel',
    'recon_image.layer_3_1.0.bias': 'RE6_GPU0/side_info_recon_model_load/layer_3_1_mainS_recon/bias',
    'recon_image.layer_3_2.0.weight': 'RE6_GPU0/side_info_recon_model_load/layer_3_2_mainS_recon/kernel',
    'recon_image.layer_3_2.0.bias': 'RE6_GPU0/side_info_recon_model_load/layer_3_2_mainS_recon/bias',
    'recon_image.layer_3_3.0.weight': 'RE6_GPU0/side_info_recon_model_load/layer_3_3_mainS_recon/kernel',
    'recon_image.layer_3_3.0.bias': 'RE6_GPU0/side_info_recon_model_load/layer_3_3_mainS_recon/bias',
    'recon_image.layer_4.1.weight': 'RE6_GPU0/side_info_recon_model_load/layer_4_mainS_recon/kernel',
    'recon_image.layer_4.1.bias': 'RE6_GPU0/side_info_recon_model_load/layer_4_mainS_recon/bias',
    'recon_image.layer_5.0.weight': 'RE6_GPU0/side_info_recon_model_load/layer_5_mainS_recon/kernel',
    'recon_image.layer_5.0.bias': 'RE6_GPU0/side_info_recon_model_load/layer_5_mainS_recon/bias',
    'recon_image.layer_6.weight': 'RE6_GPU0/side_info_recon_model_load/layer_6_mainS_recon/kernel',
    'recon_image.layer_6.bias': 'RE6_GPU0/side_info_recon_model_load/layer_6_mainS_recon/bias'
}

def convert_weight_dict(torch_d, tf_d):
  target_d = {}
  for k in torch_d:
    if not k in name_dict:
      # print('[Warning] %s not in pre-defined name dict' % (k))
      target_d[k] = torch_d[k]
      continue
    tf_k = name_dict[k]
    tf_w = np.array(tf_d[tf_k], dtype=np.float64)
    if 'kernel' in tf_k and ( 'conv' in tf_k or 'mainS_recon' in tf_k or 'transform_model' in tf_k ):
      torch_w = tf_w.transpose((3,2,0,1))
    elif ('dense' in tf_k or 'fc' in tf_k) and 'kernel' in tf_k:
      torch_w = tf_w.transpose((1,0))
    elif 'reparam_gamma' in tf_k:
      torch_w = tf_w.transpose((1,0))
    else:
      torch_w = tf_w
    target_d[k] = torch.tensor(torch_w).float()
  return target_d


class Preprocess(object):
    def __init__(self):
        pass
    def __call__(self, PIL_img):
        img = torch.from_numpy(np.asarray(PIL_img, dtype=np.float32).transpose((2, 0, 1)))
        img /= 127.5
        img -= 1.0
        return img


def unnorm(img):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    dtype = img.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=img.device)
    std = torch.as_tensor(std, dtype=dtype, device=img.device)
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    for i in range(len(img)):
        image = img[i]
        image.mul_(std).add_(mean)
        img[i] = image
    return img

def main():
    # with open('../model/model1_qp6.pk', 'rb') as f:
    #     tf_d_1 = pickle.load(f)
    resize_target = Resize_target(SCALE)
    train_transform = transforms.Compose([
        transforms.Resize(SCALE),
        # Preprocess(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))# 归一化
    ])
    train_transform = ResizeImageAndTarget(SCALE,train_transform,resize_target)
    train_data = get_coco(root=args.coco_root,image_set="train",transforms=train_transform)
    train_sampler = torch.utils.data.RandomSampler(train_data)
    batch_train_sampler = torch.utils.data.BatchSampler(train_sampler, args.batchsize, drop_last=True)
    training_loader = torch.utils.data.DataLoader(
        train_data,
        num_workers=args.num_workers,
        batch_sampler=batch_train_sampler,
        collate_fn=dataset_load_utils.utils.collate_fn,
    )
    val_transform = transforms.Compose([
        transforms.Resize(SCALE),
        # Preprocess(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # 归一化
    ])
    val_transform = ResizeImageAndTarget(SCALE, val_transform, resize_target)
    val_data = get_coco(root=args.coco_root, image_set="val", transforms=val_transform)
    val_sampler = torch.utils.data.RandomSampler(val_data)
    batch_val_sampler = torch.utils.data.BatchSampler(val_sampler, args.batchsize, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_data,
        num_workers=args.num_workers,
        batch_sampler=batch_val_sampler,
        collate_fn=dataset_load_utils.utils.collate_fn,
    )

    # In this version, parallel level should be manually set in the code.
    # The example if for training with 2 GPUs
    n_parallel = args.gpu_count
    if args.qp > 3:
        net = Net(lmbda,(args.batchsize // n_parallel, 256, 256, 3),
                  (args.batchsize // n_parallel, 256, 256, 3),args.qp)
    else:
        net = Net_low(lmbda,(args.batchsize // n_parallel, 256, 256, 3),
                      (args.batchsize // n_parallel, 256, 256, 3),args.qp)
    if bool(args.load_weights):
        sd = net.state_dict()
        with open('../model/model0_qp%d.pk'%args.qp, 'rb') as f:
            tf_d_0 = pickle.load(f)
        target_d = convert_weight_dict(sd, tf_d_0)
        target_d['get_h1_sigma'] = torch.tensor(
            np.array(tf_d_0['RE6_GPU0/get_h1_sigma/get_h1_sigma/z_sigma']).reshape((1, 32 * 4, 1, 1)))

        net.load_state_dict(target_d)

    net = nn.DataParallel(net,output_device=0)
    net.cuda()
        # opt = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.learning_rate)
    opt = optim.Adam(net.parameters(), lr=args.learning_rate)

    # for checkpoint resume
    stage = 4
    st_epoch = 0
    ###################
    logger_train = Logger(
        os.path.join(result_saved_path, 'qp'+str(args.qp)+'_log_' + str(args.batchsize) + '_' + str(args.learning_rate) + '_' + str(lmbda) + '.txt'))
    logger_val = Logger(
        os.path.join(result_saved_path,
                     'qp'+str(args.qp)+'_log_val_' + str(args.batchsize) + '_' + str(args.learning_rate) + '_' + str(lmbda) + '.txt'))
    logger_result = Logger(
        os.path.join(result_saved_path,
                     'qp'+str(args.qp)+'_log_task_' + str(args.batchsize) + '_' + str(args.learning_rate) + '_' + str(lmbda) + '.txt'))
    logger_train.set_names(
        ['Epoch', 'Train Loss', 'Train Bpp','Train Bpp1','Train Bpp2','Train Bpp3', 'Train MSE', 'Train Seg'])
    logger_val.set_names(['Epoch', 'Val Bpp','Val Bpp1','Val Bpp2','Val Bpp3', 'Val MSE','Val PSNR'])
    logger_result.set_names(['Epoch', 'APbb', 'APbb50', 'APbb75', 'APbbs', 'APbbm', 'APbbl', 'AP', 'AP50', 'AP75', 'APs', 'APm', 'APl'])
    val_acc = 0
    for epoch in range(st_epoch, 100):
        start_time = time.time()
        list_train_loss = 0.
        list_train_bpp = 0.
        list_train_mse = 0.
        list_train_seg = 0.
        list_train_bpp1 = 0.
        list_train_bpp2 = 0.
        list_train_bpp3 = 0.
        cnt = 0
        with tqdm(total=len(train_data), desc=f'Epoch train {epoch + 1}/{100}', unit='img',ncols=100) as pbar:
            for i, (images, targets) in enumerate(training_loader):

                images = torch.stack([image.cuda() for image in images], dim=0)
                targets = [{k: v.cuda() if k != 'image_id' else v for k, v in t.items()} for t in targets]
                opt.zero_grad()
                # todo：调整网络结构 组织targets，查看mask rcnn怎么训练
                train_loss, train_bpp, bpp1, bpp2, bpp3, train_mse, train_seg = net(images, targets, 'train', stage)

                train_loss = train_loss.mean()
                train_bpp = train_bpp.mean()
                train_mse = train_mse.mean()
                train_seg = train_seg.mean()
                bpp1 = bpp1.mean()
                bpp2 = bpp2.mean()
                bpp3 = bpp3.mean()

                if np.isnan(train_loss.item()):
                    raise Exception('NaN in loss')
                train_loss.backward()

                nn.utils.clip_grad_norm_(net.parameters(), 10)

                opt.step()

                list_train_loss += train_loss.item()
                list_train_bpp += train_bpp.item()
                list_train_mse += train_mse.item()
                list_train_seg += train_seg.item()
                list_train_bpp1 += bpp1.item()
                list_train_bpp2 += bpp2.item()
                list_train_bpp3 += bpp3.item()

                cnt += 1

                pbar.set_postfix(train_loss='{:.6f}'.format(train_loss.detach().cpu().numpy()))
                pbar.update(images.shape[0])
                del train_seg,train_loss
        batch_train_loss = list_train_loss / cnt
        batch_train_bpp = list_train_bpp / cnt
        batch_train_bpp1 = list_train_bpp1 / cnt
        batch_train_bpp2 = list_train_bpp2 / cnt
        batch_train_bpp3 = list_train_bpp3 / cnt
        batch_train_mse = list_train_mse / cnt
        batch_train_seg = list_train_seg / cnt
        timestamp = time.time()
        print('[Epoch %04d TRAIN %.1f seconds] Loss: %.4e bpp: %.4e bpp1: %.4e bpp2: %.4e bpp3: %.4e mse: %.4e seg: %.4e' % (
            epoch,timestamp - start_time,batch_train_loss,batch_train_bpp,
            batch_train_bpp1,batch_train_bpp2,batch_train_bpp3,
            batch_train_mse, batch_train_seg))
        batch_val_bpp, batch_val_mse, batch_val_psnr, batch_val_bpp1, batch_val_bpp2, batch_val_bpp3, eval_result = test(net,epoch,stage,val_data,val_loader,val_transform)
        if val_acc < float(eval_result['bbox'][0])+float(eval_result['segm'][0]):
            val_acc = float(eval_result['bbox'][0]) + float(eval_result['segm'][0])
            print('[INFO] Saving')
            if not os.path.isdir(args.checkpoint_dir):
                os.mkdir(args.checkpoint_dir)
            torch.save(net.state_dict(), './%s/qp%d_%04d.ckpt' %
                       (args.checkpoint_dir,args.qp, epoch))
            torch.save(opt.state_dict(), './%s/latest_opt_qp%d.ckpt' % (args.checkpoint_dir,args.qp))
        # todo 保存训练数据
        logger_train.append(
            [epoch, batch_train_loss, batch_train_bpp,batch_train_bpp1,batch_train_bpp2,batch_train_bpp3, batch_train_mse, batch_train_seg])
        logger_val.append([epoch, batch_val_bpp, batch_val_bpp1, batch_val_bpp2,batch_val_bpp3,batch_val_mse,batch_val_psnr])
        logger_result.append([epoch, eval_result['bbox'][0], eval_result['bbox'][1], eval_result['bbox'][2],
                              eval_result['bbox'][3],eval_result['bbox'][4],eval_result['bbox'][5],
                              eval_result['segm'][0],eval_result['segm'][1],eval_result['segm'][2],eval_result['segm'][3],
                              eval_result['segm'][4],eval_result['segm'][5]])
        # sch.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "output", nargs="?",
        help="Output filename.")
    parser.add_argument(
        "--coco_root", default='../../../coco_dataset', type=str,
        help='coco dataset root path')
    parser.add_argument(
        "--checkpoint_dir", default="train",
        help="Directory where to save/load model checkpoints.")
    parser.add_argument(
        "--batchsize", type=int, default=18,
        help="Batch size for training.")
    parser.add_argument(
        '--gpu', default='0,3', type=str, help='gpu id')
    parser.add_argument(
        "--gpu_count", type=int, default=2,
        help="gpu count")
    parser.add_argument(
        "--qp", type=int, default=3,
        help="quantization parameter")
    parser.add_argument(
        "--num_workers", type=int, default=16,
        help="num workers for data loading.")
    parser.add_argument(
        "--patchsize", type=int, default=256,
        help="Size of image patches for training.")
    parser.add_argument(
        "--learning_rate", type=int, default=0.0001,
        help="learning rate")
    parser.add_argument(
        "--preprocess_threads", type=int, default=1,
        help="Number of CPU threads to use for parallel decoding of training "
             "images.")
    parser.add_argument(
        "--load_weights", default=1,
        help="Loaded weights")

    args = parser.parse_args()

    lmbda = {"1":0.0012,"2":0.0015,"3":0.0025,"4":0.008,"5":0.015,"6":0.02,"7":0.03}[str(args.qp)]
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    main()

