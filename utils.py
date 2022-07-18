from matplotlib import pyplot as plt
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import torch

def cmap_mapping(disp_resized, cmap = 'jet', max = None):
    """Rescale image pixels to span range [0, 1]
    """
    disp_resized_np = disp_resized.squeeze().detach().cpu().numpy()
    if max is not None:
        normalizer = mpl.colors.Normalize(vmin=0.1, vmax=max)
    else:
        vmax = np.percentile(disp_resized_np, 95)
        vmin = disp_resized_np.min()
        normalizer = mpl.colors.Normalize(vmin= vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap= cmap)
    colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
    colormapped_im = colormapped_im.transpose((2,0,1))
    return torch.from_numpy(colormapped_im)

def ref_cmap_mapping(disp_resized, ref, cmap = 'jet', max = None):
    """Rescale image pixels to span range [0, 1]
    """
    disp_resized_np = disp_resized.squeeze().detach().cpu().numpy()
    ref_np = ref.squeeze().detach().cpu().numpy()
    vmax = np.percentile(ref_np, 95)
    if max is not None:
        normalizer = mpl.colors.Normalize(vmin=0.1, vmax=max)
    else:
        normalizer = mpl.colors.Normalize(vmin=ref_np.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap= cmap)
    colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
    colormapped_im = colormapped_im.transpose((2,0,1))
    return torch.from_numpy(colormapped_im)    

def err_map(gt, pred):
    """Rescale image pixels to span range [0, 1]
    """

    gt = gt.squeeze().detach().cpu().numpy()
    pred = pred.squeeze().detach().cpu().numpy()
    diff = np.abs(gt-pred).mean(0) #.transpose((1,2,0))
    normalizer = mpl.colors.Normalize(vmin=0, vmax=1.0)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='jet')
    colormapped_im = (mapper.to_rgba(diff)[:, :, :3] * 255).astype(np.uint8)
    colormapped_im = colormapped_im.transpose((2,0,1))
    return torch.from_numpy(colormapped_im) 

def err_map2(img):
    """Rescale image pixels to span range [0, 1]
    """
    img = img.squeeze().detach().cpu().numpy()
    normalizer = mpl.colors.Normalize(vmin=0, vmax=1.0)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='jet')
    colormapped_im = (mapper.to_rgba(img)[:, :, :3] * 255).astype(np.uint8)
    colormapped_im = colormapped_im.transpose((2,0,1))
    return torch.from_numpy(colormapped_im) 


def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def normalize_image(x):
    """Rescale image pixels to span range [0, 1]
    """
    ma = float(x.max().cpu().data)
    mi = float(x.min().cpu().data)
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d


def sec_to_hm(t):
    """Convert time in seconds to time in hours, minutes and seconds
    e.g. 10239 -> (2, 50, 39)
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return t, m, s


def sec_to_hm_str(t):
    """Convert time in seconds to a nice string
    e.g. 10239 -> '02h50m39s'
    """
    h, m, s = sec_to_hm(t)
    return "{:02d}h{:02d}m{:02d}s".format(h, m, s)


def disparity_regression(x, maxdisp):
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=True)


def L1_mask(gt, pred, mask, return_map = False):
    # compute L1 loss on the valid pixels of gt
    m = mask > 0
    loss = torch.abs(gt[m] - pred[m])
    if return_map:
        return loss.mean(), torch.abs(gt - pred) * m 
    else: 
        return loss.mean()

## Reference : https://github.com/zzangjinsun/NLSPN_ECCV20
def evaluate(gt_, pred_, loss, is_test = False):
    t_valid = 0.0001
    metric_name = ['1.RMSE', '3.MAE', '4.iRMSE', '5.iMAE', '2.REL', 'D1', 'D2', 'D3']
    with torch.no_grad():
        pred = pred_.detach()
        gt = gt_.detach()

        pred_inv = 1.0 / (pred + 1e-8)
        gt_inv = 1.0 / (gt + 1e-8)

        # For numerical stability
        mask = gt > t_valid
        num_valid = mask.sum()

        pred = pred[mask]
        gt = gt[mask]

        pred_inv = pred_inv[mask]
        gt_inv = gt_inv[mask]

        pred_inv[pred <= t_valid] = 0.0
        gt_inv[gt <= t_valid] = 0.0

        # RMSE / MAE
        diff = pred - gt
        diff_abs = torch.abs(diff)
        diff_sqr = torch.pow(diff, 2)

        rmse = diff_sqr.sum() / (num_valid + 1e-8)
        rmse = torch.sqrt(rmse)

        mae = diff_abs.sum() / (num_valid + 1e-8)

        # iRMSE / iMAE
        diff_inv = pred_inv - gt_inv
        diff_inv_abs = torch.abs(diff_inv)
        diff_inv_sqr = torch.pow(diff_inv, 2)

        irmse = diff_inv_sqr.sum() / (num_valid + 1e-8)
        irmse = torch.sqrt(irmse)

        imae = diff_inv_abs.sum() / (num_valid + 1e-8)

        # Rel
        rel = diff_abs / (gt + 1e-8)
        rel = rel.sum() / (num_valid + 1e-8)

        # delta
        r1 = gt / (pred + 1e-8)
        r2 = pred / (gt + 1e-8)
        ratio = torch.max(r1, r2)

        del_1 = (ratio < 1.25).type_as(ratio)
        del_2 = (ratio < 1.25**2).type_as(ratio)
        del_3 = (ratio < 1.25**3).type_as(ratio)

        del_1 = del_1.sum() / (num_valid + 1e-8)
        del_2 = del_2.sum() / (num_valid + 1e-8)
        del_3 = del_3.sum() / (num_valid + 1e-8)

        result = [rmse, mae, irmse, imae, rel, del_1, del_2, del_3]
        if is_test:
            result = torch.stack(result)
            result = torch.unsqueeze(result, dim=0).detach()
            return result
        # for i in range(len(metric_name)):
        #     loss[metric_name[i]] = result[i].detach()
        
        for i in range(len(result)):
            name = "Eval_/{}".format(metric_name[i])
            loss[name] = result[i].detach()

        return loss