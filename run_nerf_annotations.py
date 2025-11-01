import os
import torch
# torch.autograd.set_detect_anomaly(True)  # 用于检测梯度异常，调试时开启
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, imageio
import time
from tqdm import tqdm, trange


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEBUG = False


# Misc 杂项功能函数
img2mse = lambda x, y : torch.mean((x - y) ** 2)  # 计算均方误差
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))  # 将MSE转换为PSNR指标
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)  # 将浮点数图像转换为8位图像

# Positional encoding (section 5.1) 位置编码类
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []  # 存储编码函数列表
        d = self.kwargs['input_dims']  # 输入维度
        out_dim = 0  # 输出维度
        
        # 是否包含原始输入
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']  # 最大频率对数
        N_freqs = self.kwargs['num_freqs']  # 频率数量
        
        # 对数采样或线性采样
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        # 为每个频率创建sin和cos编码函数
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        # 应用所有编码函数并拼接结果
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    """获取位置编码器"""
    if i == -1:
        return nn.Identity(), 3 #恒等映射
    
    # 编码器配置参数
    embed_kwargs = {
    'include_input': True,      # 是否包含原始输入，保留低频信息，基本都为 True
    'input_dims': 3,           # 输入维度（3D坐标）--》xyz
    'max_freq_log2': multires-1, # 最大频率的log2值
    'num_freqs': multires,     # 频率数量
    'log_sampling': True,      # 对数尺度采样频率  #frequencies = [2^0, 2^1, 2^2, 2^3, 2^4] = [1, 2, 4, 8, 16]
    'periodic_fns': [torch.sin, torch.cos],  # 使用的周期函数
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


# Model NeRF模型定义
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """ 
        D: 网络深度
        W: 网络宽度
        input_ch: 输入通道数（位置编码后）
        input_ch_views: 视角输入通道数
        output_ch: 输出通道数
        skips: 跳跃连接层位置
        use_viewdirs: 是否使用视角方向
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        # 位置编码处理网络
        self.pts_linears = nn.ModuleList(   #点云输入处理层
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        # 视角方向处理网络
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        # 输出层
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)  # 密度输出
            self.rgb_linear = nn.Linear(W//2, 3)  #三维rgb
        else:
            self.output_linear = nn.Linear(W, output_ch)

        # 调试信息：打印网络结构
        print(f"[NeRF Debug] Network initialized:")
        print(f"  - Depth: {D}, Width: {W}")
        print(f"  - Input channels: {input_ch}, View channels: {input_ch_views}")
        print(f"  - Output channels: {output_ch}")
        print(f"  - Skip connections at layers: {skips}")
        print(f"  - Use view directions: {use_viewdirs}")
        print(f"  - Points linears layers: {len(self.pts_linears)}")
        print(f"  - Views linears layers: {len(self.views_linears)}")

    def forward(self, x):
        # 调试信息：输入形状
        print(f"[NeRF Forward Debug] Input shape: {x.shape}")
        
        # 分割输入为位置和视角
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        print(f"[NeRF Forward Debug] Points shape: {input_pts.shape}, Views shape: {input_views.shape}")
        
        h = input_pts
        
        # 位置编码网络前向传播
        for i, l in enumerate(self.pts_linears):
            h_prev = h.shape
            h = self.pts_linears[i](h)
            h = F.relu(h)
            print(f"[NeRF Forward Debug] Layer {i}: {h_prev} -> {h.shape}")
            
            if i in self.skips:  # 跳跃连接
                h = torch.cat([input_pts, h], -1)
                print(f"[NeRF Forward Debug] Skip connection at layer {i}: concatenated shape {h.shape}")

        # 根据是否使用视角方向处理输出
        if self.use_viewdirs:
            alpha = self.alpha_linear(h)  #输出密度值一维
            print(f"[NeRF Forward Debug] Alpha shape: {alpha.shape}")
            
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
            print(f"[NeRF Forward Debug] After feature concat: {h.shape}")
        
            # 视角方向网络前向传播
            for i, l in enumerate(self.views_linears):
                h_prev = h.shape
                h = self.views_linears[i](h)
                h = F.relu(h)
                print(f"[NeRF Forward Debug] View layer {i}: {h_prev} -> {h.shape}")

            rgb = self.rgb_linear(h)  # RGB颜色输出
            print(f"[NeRF Forward Debug] RGB shape: {rgb.shape}")
            
            outputs = torch.cat([rgb, alpha], -1)  # 拼接RGB和密度
            print(f"[NeRF Forward Debug] Final output shape: {outputs.shape}")
        else:
            outputs = self.output_linear(h)
            print(f"[NeRF Forward Debug] Final output shape: {outputs.shape}")

        return outputs  

def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret

def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    print(f"[run_network Debug] Inputs shape: {inputs.shape}")
    print(f"[run_network Debug] Viewdirs shape: {viewdirs.shape if viewdirs is not None else 'None'}")
    
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    print(f"[run_network Debug] Inputs flattened shape: {inputs_flat.shape}，每一个 ray 都有 64 的 sample 点，用 flat 做展平")
    
    embedded = embed_fn(inputs_flat)
    print(f"[run_network Debug] After positional encoding shape: {embedded.shape}")

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)  #需要扩展方向信息以匹配采样点数量
        print(f"[run_network Debug] Expanded viewdirs shape: {input_dirs.shape}")
        
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        print(f"[run_network Debug] Viewdirs flattened shape: {input_dirs_flat.shape}")
        
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        print(f"[run_network Debug] Viewdirs after encoding shape: {embedded_dirs.shape}")
        
        embedded = torch.cat([embedded, embedded_dirs], -1)
        print(f"[run_network Debug] After concatenation shape: {embedded.shape}")

    print(f"[run_network Debug] Using netchunk: {netchunk}")
    print('-'*50)
    outputs_flat = batchify(fn, netchunk)(embedded)
    print(f"[run_network Debug] Network outputs flat shape: {outputs_flat.shape}")
    
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])#恢复形状
    print(f"[run_network Debug] Final outputs shape: {outputs.shape}")
    
    return outputs

def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True):
    print("\n=== [Step 1] 加载 poses_bounds.npy ===")
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    print(f"poses_arr.shape: {poses_arr.shape}")  # (N, 3*5+2)
    print("-" * 50)

    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
    print(f"poses after reshape and transpose shape: {poses.shape}")
    
    bds = poses_arr[:, -2:].transpose([1, 0])
    print(f"bds after transpose shape: {bds.shape}")
    
    print("=== [Step 2] 拆分 poses / bds ===")
    print(f"poses.shape: {poses.shape}  (应为 [3,5,N])")
    print(f"bds.shape: {bds.shape}  (应为 [2,N])")
    print("-" * 50)

    img0 = [os.path.join(basedir, 'images', f)
            for f in sorted(os.listdir(os.path.join(basedir, 'images')))
            if f.endswith(('JPG', 'jpg', 'png'))][0]
    sh = imageio.imread(img0).shape
    print("=== [Step 3] 检查原始图像尺寸 ===")
    print(f"Sample image path: {img0}")
    print(f"Image shape: {sh}")
    print("-" * 50)

    sfx = ''
    print('-'*40)
    print(factor, height, width)
    if factor is not None:
        sfx = f'_{factor}'
        _minify(basedir, factors=[factor])
        print(f"应用 factor 缩放: {factor}")
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = f'_{width}x{height}'
        print(f"应用分辨率缩放: {width}x{height}")
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = f'_{width}x{height}'
        print(f"应用分辨率缩放: {width}x{height}")
    else:
        factor = 1
        print("未指定缩放参数，使用原图大小。")

    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print(imgdir, 'does not exist, returning')
        return

    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))
                if f.endswith(('JPG', 'jpg', 'png'))]
    print("=== [Step 4] 图像文件匹配 ===")
    print(f"图像数量: {len(imgfiles)}")
    print(f"poses 数量: {poses.shape[-1]}")
    if poses.shape[-1] != len(imgfiles):
        print(f"!! 图像数和姿态数不匹配: imgs={len(imgfiles)}, poses={poses.shape[-1]}")
        return
    print("-" * 50)

    sh = imageio.imread(imgfiles[0]).shape
    print(f"Sample image shape after minify: {sh}")
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    print(f"poses after updating H,W shape: {poses.shape}")
    
    poses[2, 4, :] = poses[2, 4, :] * 1. / factor
    print(f"poses after updating focal shape: {poses.shape}")

    print("=== [Step 5] 更新 H, W, focal ===")
    print(f"H={sh[0]}, W={sh[1]}, factor={factor}")
    print("-" * 50)

    if not load_imgs:
        print("未加载图像，仅返回 poses/bds。")
        return poses, bds

    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f)
        else:
            return imageio.imread(f)

    imgs = [imread(f)[..., :3] / 255. for f in imgfiles]
    print(f"imgs list length: {len(imgs)}")
    print(f"first image shape: {imgs[0].shape}")
    
    imgs = np.stack(imgs, -1)
    print(f"imgs after stack shape: {imgs.shape}")

    print("=== [Step 6] 图像堆叠完成 ===")
    print(f"imgs.shape: {imgs.shape} (应为 [H, W, 3, N])")
    print("=" * 60 + "\n")

    return poses, bds, imgs

def load_llff_data(basedir, factor=8, recenter=True, bd_factor=.75, spherify=True, path_zflat=False):
    
    poses, bds, imgs = _load_data(basedir, factor=factor)
    
    print("=== 加载完成 ===")
    print(f"poses shape(before transform): {poses.shape}")
    print(f"bds shape(before moveaxis): {bds.shape}")
    print(f"imgs shape(before moveaxis): {imgs.shape}")
    print("-" * 50)
    
    # Correct rotation matrix ordering and move variable dim to axis 0
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    print(f"poses after concat shape: {poses.shape}")
    
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    print(f"poses after moveaxis shape: {poses.shape}")
    
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    print(f"imgs after moveaxis shape: {imgs.shape}")
    
    images = imgs
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)
    print(f"bds after moveaxis shape: {bds.shape}")
    
    print("=== 坐标系转换后 ===")
    print(f"poses shape(after moveaxis): {poses.shape}")
    print(f"images shape(after moveaxis): {images.shape}")
    print(f"bds shape(after moveaxis): {bds.shape}")
    print("-" * 50)
    
    # Rescale if bd_factor is provided
    sc = 1. if bd_factor is None else 1./(bds.min() * bd_factor)
    poses[:,:3,3] *= sc
    bds *= sc
    
    print("=== 缩放后 ===")
    print(f"scale factor (sc): {sc:.6f}")
    print(f"poses translation range: {poses[:, :3, 3].min():.4f} ~ {poses[:, :3, 3].max():.4f}")
    print(f"bds range: {bds.min():.4f} ~ {bds.max():.4f}")
    print("-" * 50)

    if recenter:
        print("=== 重新中心化 ===")
        poses = recenter_poses(poses)
        print("Recentered poses.", poses.shape)
        print("-" * 50)

    if spherify:
        print("=== 球形化相机轨迹 ===")
        poses, render_poses, bds = spherify_poses(poses, bds)
        print("Spherified poses.")
        print(f"render_poses shape: {render_poses.shape}")
        print("-" * 50)

    print("=== 最终输出 ===")
    print(f"最终 poses shape: {poses.shape}")
    print(f"最终 images shape: {images.shape}")
    print(f"最终 bds shape: {bds.shape}")
    print("=" * 60)


    render_poses = np.array(render_poses).astype(np.float32)

    c2w = poses_avg(poses)
    print('Data:')
    print(poses.shape, images.shape, bds.shape)
    
    dists = np.sum(np.square(c2w[:3,3] - poses[:,:3,3]), -1) #相机位置与平均位置之间的 距离平方。
    i_test = np.argmin(dists)  #找到最小，作为测试相机
    print('HOLDOUT view is', i_test)
    
    images = images.astype(np.float32)
    poses = poses.astype(np.float32)
    print('----------------------------')

    return images, poses, bds, render_poses, i_test

def _minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return

def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def poses_avg(poses):
    print(f"=== poses_avg 输入 ===")
    print(f"poses shape: {poses.shape}")
    
    hwf = poses[0, :3, -1:] #最后一列hwf
    print(f"hwf shape: {hwf.shape}")

    center = poses[:, :3, 3].mean(0)#所有相机平移向量均值
    vec2 = normalize(poses[:, :3, 2].sum(0))  #对所有相机的前向轴（z）进行求和并归一化，得到单位向量
    up = poses[:, :3, 1].sum(0)  #对所有相机的上向（y）轴进行求和
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    
    print(f"=== poses_avg 输出 ===")
    print(f"c2w shape: {c2w.shape}")
    return c2w

def recenter_poses(poses):
    print(f"=== recenter_poses 输入 ===")
    print(f"poses shape: {poses.shape}")
    
    poses_ = poses+0
    bottom = np.reshape([0,0,0,1.], [1,4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)
    #标准4*4_homogeneous coordinates
    print(f"c2w after concat shape: {c2w.shape}")
    
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])# tile--》在各个维度上可重复扩展bo
    print(f"bottom after tile shape: {bottom.shape}")
    
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)
    print(f"poses after concat shape: {poses.shape}")

    poses = np.linalg.inv(c2w) @ poses  #mapping from world to average camera
    print(f"poses after inv @ shape: {poses.shape}")
    
    poses_[:,:3,:4] = poses[:,:3,:4]
    print(f"poses_ after assignment shape: {poses_.shape}")
    poses = poses_
    print(f"poses final shape: {poses.shape}")  #new的位置信息已被覆盖到原 n*3*5 的 poses 中
    
    print(f"=== recenter_poses 输出 ===")
    print(f"poses shape: {poses.shape}")
    return poses

def spherify_poses(poses, bds):
    print(f"=== spherify_poses 输入 ===")
    print(f"poses shape: {poses.shape}, bds shape: {bds.shape}")
    
    p34_to_44 = lambda p : np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1,:], [1,1,4]), [p.shape[0], 1,1])], 1)
    
    rays_d = poses[:,:3,2:3]  #z-axis ，相机朝向
    rays_o = poses[:,:3,3:4]  #相机位置
    print(f"rays_d shape: {rays_d.shape}, rays_o shape: {rays_o.shape}")

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0,2,1])  #rays_d方向上的正交投影  单位矩阵减去平行投影
        b_i = -A_i @ rays_o #射线原点在与射线方向垂直的平面上的投影。
        pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0,2,1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)
    print(f"pt_mindist shape: {pt_mindist.shape}",pt_mindist)
    #center是一个坐标：pt_mindist shape: (3,) [ 1.95008822e-03 -5.59532012e-04 -2.51793740e+00]
    
    center = pt_mindist
    
    up = (poses[:,:3,3] - center).mean(0)  #相机群的平均偏移量（vector）
    print(f"up shape: {up.shape}")

    vec0 = normalize(up)
    vec1 = normalize(np.cross([.1,.2,.3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)
    print(f"c2w shape: {c2w.shape}")

    print('添加维度？',c2w[None].shape)
    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:,:3,:4])  
    print(f"poses_reset shape: {poses_reset.shape}")

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:,:3,3]), -1)))  #平均半径
    
    sc = 1./rad
    poses_reset[:,:3,3] *= sc
    bds *= sc
    rad *= sc
    
    centroid = np.mean(poses_reset[:,:3,3], 0)
    print(f"centroid shape: {centroid.shape}")
    zh = centroid[2]
    radcircle = np.sqrt(rad**2-zh**2)
    new_poses = []
    
    for th in np.linspace(0.,2.*np.pi, 120):
        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0,0,-1.])

        vec2 = normalize(camorigin)  
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)
        new_poses.append(p)

    print(f"new_poses list length: {new_poses.__len__()}")

    new_poses = np.stack(new_poses, 0)
    print(f"new_poses after stack shape: {new_poses.shape}")
    
    new_poses = np.concatenate([new_poses, np.broadcast_to(poses[0,:3,-1:], new_poses[:,:3,-1:].shape)], -1)
    print(f"new_poses after concat shape: {new_poses.shape}")
    
    poses_reset = np.concatenate([poses_reset[:,:3,:4], np.broadcast_to(poses[0,:3,-1:], poses_reset[:,:3,-1:].shape)], -1)
    print(f"poses_reset after concat shape: {poses_reset.shape}")
    
    print(f"=== spherify_poses 输出 ===")
    print(f"poses_reset shape: {poses_reset.shape}, new_poses shape: {new_poses.shape}, bds shape: {bds.shape}")
    return poses_reset, new_poses, bds

def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    print(f"[raw2outputs Debug] raw shape: {raw.shape}")
    print(f"[raw2outputs Debug] z_vals shape: {z_vals.shape}")
    print(f"[raw2outputs Debug] rays_d shape: {rays_d.shape}")
    print(f"[raw2outputs Debug] raw_noise_std: {raw_noise_std}")
    print(f"[raw2outputs Debug] white_bkgd: {white_bkgd}")
    print(f"[raw2outputs Debug] pytest: {pytest}")
    print("-"*50)
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)#α = 1 - exp(-σ * δ)

    dists = z_vals[...,1:] - z_vals[...,:-1]#计算相邻采样点之间的距离
    dists=dists.to(device)
    print(dists.device,dists.shape)
    dists = torch.cat([dists, (torch.Tensor([1e10]).expand(dists[...,:1].shape)).to(device)], -1)  # [N_rays, N_samples]
    print(f"[raw2outputs Debug] dists after cat shape: {dists.shape}")


    a=torch.norm(rays_d[...,None,:], dim=-1)
    print(a.device,a.shape)
    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)#每一个射线的所有采样点都乘以该射线对应的模长
    print(f"[raw2outputs Debug] dists after scaling shape: {dists.shape}")

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]  通过 sigmoid 函数映射到 [0,1] 范围。
    print(f"[raw2outputs Debug] rgb shape: {rgb.shape}")

    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    print(f"[raw2outputs Debug] alpha shape: {alpha.shape}")

    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    acc_bound=torch.ones((alpha.shape[0], 1))
    acc_bound=acc_bound.to(device)
    acc_rate=torch.cumprod(torch.cat([acc_bound, 1.-alpha + 1e-10], -1), -1)
    acc_rate=acc_rate.to(device)
    weights = alpha * acc_rate[:, :-1]#device

    print(f"[raw2outputs Debug] weights shape: {weights.shape}")
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]
    print(f"[raw2outputs Debug] rgb_map shape: {rgb_map.shape}")

    depth_map = torch.sum(weights * z_vals, -1) #光线终止深度的期望值
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))#归一化考虑穿过光线
    acc_map = torch.sum(weights, -1)  #射线击中物体概率

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map

def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf

    weights = weights + 1e-5 #数值稳定性：添加小常数防止除零错误和NaN值
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)#累加sum
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))
    print('sample_pdf 调用信息:------------------')
    print('bins',bins.device,bins.shape)
    print('pdf',pdf.device,pdf.shape)
    print('cdf',cdf.device,cdf.shape)

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
        print('u',u.device,u.shape)
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])
        print(u.device,u.shape)

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    u=u.to(device)


    inds = torch.searchsorted(cdf, u, right=True)    #  找到u 值在 cdf 中对应的右边界位置
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    print('below',below.device,below.shape)
    print('above',above.device,above.shape)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)
    print('inds_g',inds_g.device,inds_g.shape)
    

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    print('matched_shape',matched_shape)
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)
    print('cdf_g',cdf_g.device,cdf_g.shape)
    print('bins_g',bins_g.device,bins_g.shape)
    #获得采样点对应概率区间


    #重新做密集采样


    denom = (cdf_g[...,1]-cdf_g[...,0])  #区间宽度
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)#如果区间宽度太小（接近0），用1替换
    t = (u-cdf_g[...,0])/denom#在概率区间内的相对位置
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])
    print('samples',samples.device,samples.shape)

    return samples

def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    print(f"渲染光线批次信息:")
    print(f"  - 光线数量: {N_rays}")
    print(f"  - rays_o 设备: {rays_o.device}, 形状: {rays_o.shape}")
    print(f"  - rays_d 设备: {rays_d.device}, 形状: {rays_d.shape}")
    print(f"  - viewdirs 设备: {viewdirs.device if viewdirs is not None else 'None'}",viewdirs.shape if viewdirs is not None else '')
    print(f"  - near 设备: {near.device}, far 设备: {far.device}")
    print(f"  - N_samples: {N_samples}, N_importance: {N_importance}")

    t_vals = torch.linspace(0., 1., steps=N_samples)
    t_vals=t_vals.to(device)
    print('lindisp',lindisp)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)  
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    print(f"采样点信息:")
    print(f"  - z_vals 设备: {z_vals.device}, 形状: {z_vals.shape}")

    if perturb > 0.:#无扰动
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand
        print(f"  - 扰动后 z_vals 设备: {z_vals.device}, 形状: {z_vals.shape}")

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]# [N_rays, N_samples, 3]
    print(f"采样点 pts 信息:",pts.shape)
    print('-'*50)
    print(f"  - 采样点 pts 设备: {pts.device}, 形状: {pts.shape}")

#     raw = run_network(pts)
    print("开始 coarse 网络查询...")
    raw = network_query_fn(pts, viewdirs, network_fn)#这里面的 network_fn 就是传进来的model--class-nerf
    print(f"coarse 网络输出 raw 设备: {raw.device}, 形状: {raw.shape}")
    
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)  #device
    print(f"coarse 输出:")
    print(f"  - rgb_map 设备: {rgb_map.device}, 形状: {rgb_map.shape}")
    print(f"  - disp_map 设备: {disp_map.device}, 形状: {disp_map.shape}")
    print(f"  - acc_map 设备: {acc_map.device}, 形状: {acc_map.shape}")
    print(f"  - weights 设备: {weights.device}, 形状: {weights.shape}")

    if N_importance > 0:

        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()#采样结果从计算图中分离，避免梯度传播到采样操作

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]
        print('-'*50)
        print(f"fine 网络采样点:")
        print(f"  - z_samples 设备: {z_samples.device}, 形状: {z_samples.shape}")
        print(f"  - 合并后 z_vals 设备: {z_vals.device}, 形状: {z_vals.shape}")
        print(f"  - fine 采样点 pts 设备: {pts.device}, 形状: {pts.shape}")


        run_fn = network_fn if network_fine is None else network_fine
#         raw = run_network(pts, fn=run_fn)
        print("开始 fine 网络查询...")
        raw = network_query_fn(pts, viewdirs, run_fn)
        print(f"fine 网络输出 raw 设备: {raw.device}, 形状: {raw.shape}")

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)
        print(f"fine 输出:")
        print(f"  - rgb_map 设备: {rgb_map.device}, 形状: {rgb_map.shape}")
        print(f"  - disp_map 设备: {disp_map.device}, 形状: {disp_map.shape}")
        print(f"  - acc_map 设备: {acc_map.device}, 形状: {acc_map.shape}")

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    # 打印最终返回结果的设备信息
    print("最终返回结果设备信息:")
    for k in ret:
        if isinstance(ret[k], torch.Tensor):
            print(f"  - {k}: 设备 {ret[k].device}, 形状 {ret[k].shape}, 类型 {ret[k].dtype}")
        else:
            print(f"  - {k}: {type(ret[k])}")

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")


    return ret

def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret

# Ray helpers
def get_rays(H, W, K, c2w):     #把二维像素坐标 (i, j) 映射成三维世界空间中的射线 (ray)，即每个像素对应一条光线。
    print(f"[get_rays Debug] === 开始 get_rays ===")
    print(f"[get_rays Debug] 输入参数:")
    print(f"  H: {H}, W: {W} (类型: {type(H)}, {type(W)})")
    print(f"  K shape: {K.shape}, K device: {K.device if hasattr(K, 'device') else 'numpy array'}")
    print(f"  c2w shape: {c2w.shape}, c2w device: {c2w.device}")
    print(f"  c2w type: {type(c2w)}")
    
    # 创建网格
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H), indexing='ij')  #i，j分别存储每个像素的x，y坐标，都是W*H大小
    print(f"[get_rays Debug] 网格创建:")
    print(f"  i shape: {i.shape}, i device: {i.device}")
    print(f"  j shape: {j.shape}, j device: {j.device}")
    
    i = i.t()
    j = j.t()
    print(f"[get_rays Debug] 转置后:")
    print(f"  i shape: {i.shape}, j shape: {j.shape}")
    
    # 计算射线方向（相机坐标系）
    print(f"[get_rays Debug] 计算射线方向:")
    print(f"  K[0][2] type: {type(K[0][2])}, value: {K[0][2]}")
    print(f"  K[1][2] type: {type(K[1][2])}, value: {K[1][2]}")
    print(f"  K[0][0] type: {type(K[0][0])}, value: {K[0][0]}")
    print(f"  K[1][1] type: {type(K[1][1])}, value: {K[1][1]}")
    
    # 确保所有张量在同一个设备上
    if isinstance(K, torch.Tensor):
        K_device = K.device
    else:
        K_device = device  # 如果是numpy数组，使用默认设备
    
    dirs = torch.stack([
        (i - K[0][2].item() if not isinstance(K[0][2], torch.Tensor) else (i - K[0][2].to(i.device))) / (K[0][0].item() if not isinstance(K[0][0], torch.Tensor) else K[0][0].to(i.device)),
        -(j - K[1][2].item() if not isinstance(K[1][2], torch.Tensor) else (j - K[1][2].to(j.device))) / (K[1][1].item() if not isinstance(K[1][1], torch.Tensor) else K[1][1].to(j.device)),
        -torch.ones_like(i)
    ], -1)
    
    print(f"[get_rays Debug] 射线方向计算完成:")
    print(f"  dirs shape: {dirs.shape}, dirs device: {dirs.device}")        #([378, 504, 3]) 对应每一个 pixel 的方向向量
    
    # 旋转射线方向从相机坐标系到世界坐标系
    print(f"[get_rays Debug] 旋转射线方向:")
    print(f"  c2w shape: {c2w.shape}, c2w device: {c2w.device}")
    print(f"  c2w[:3,:3] shape: {c2w[:3,:3].shape}, device: {c2w[:3,:3].device}")
    
    # 确保dirs和c2w在同一个设备上
    if dirs.device != c2w.device:
        print(f"[get_rays Debug] WARNING: 设备不匹配! dirs在{dirs.device}, c2w在{c2w.device}")
        print(f"[get_rays Debug] 将dirs移动到{c2w.device}")
        dirs = dirs.to(c2w.device)
        print(f"[get_rays Debug] dirs移动后 device: {dirs.device}")
    
    # 扩展dirs维度以进行矩阵乘法
    dirs_expanded = dirs[..., np.newaxis, :]  # [H, W, 1, 3]   插在倒数第二位   对于 h,w 位置的像素，有一个 1*3 的向量来@3*3矩阵
    print(f"[get_rays Debug] dirs扩展后: {dirs_expanded.shape}, device: {dirs_expanded.device}")

    
    # 执行矩阵乘法：将射线方向从相机坐标系转换到世界坐标系
    rays_d = torch.sum(dirs_expanded * c2w[:3,:3], -1)  # [H, W, 3]
    print(f"[get_rays Debug] 射线方向转换完成:")
    print(f"  rays_d shape: {rays_d.shape}, device: {rays_d.device}")
    #更准确地说：dir是在相机坐标系下表达的方向向量，而 NeRF 的世界是以“世界坐标系”为标准，所以要把dir从相机坐标系变换到世界坐标系。这个变换由矩阵c2w完成。

    # 计算射线起点（相机位置在世界坐标系中）
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    print(f"[get_rays Debug] 射线起点:")
    print(f"  rays_o shape: {rays_o.shape}, device: {rays_o.device}")
    print(f"[get_rays Debug] === get_rays 完成 ===\n")
    
    return rays_o, rays_d

def ndc_rays(H, W, focal, near, rays_o, rays_d):
    print(f"[ndc_rays Debug] === 开始 ndc_rays ===")
    print(f"[ndc_rays Debug] 输入参数:")
    print(f"  H: {H}, W: {W}, focal: {focal}")
    print(f"  near: {near}")
    print(f"  rays_o shape: {rays_o.shape}, device: {rays_o.device}")
    print(f"  rays_d shape: {rays_d.shape}, device: {rays_d.device}")
    
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    #沿射线方向移动原点，直到它与 z = -near（近裁剪平面）相交。
    
    # Projection
    #投影变换（核心透视映射）
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]
     
    #射线方向的投影变换
    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    print(f"[ndc_rays Debug] 输出:")
    print(f"  rays_o shape: {rays_o.shape}, device: {rays_o.device}")
    print(f"  rays_d shape: {rays_d.shape}, device: {rays_d.device}")
    print(f"[ndc_rays Debug] === ndc_rays 完成 ===\n")
    
    return rays_o, rays_d

def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d
    


def render(H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    print('-'*40)
    print(f"[render Debug] === 开始 render ===")
    print(f"[render Debug] 输入参数:")
    print(f"  H: {H}, W: {W}")
    print(f"  K shape: {K.shape}, K type: {type(K)}")
    print(f"  c2w: {c2w is not None}, c2w device: {c2w.device if c2w is not None else 'None'}")
    print(f"  ndc: {ndc}, near: {near}, far: {far}")
    print(f"  use_viewdirs: {use_viewdirs}")
    
    if c2w is not None:
        # special case to render full image
        print(f"[render Debug] 使用c2w生成光线")
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        print(f"[render Debug] 使用提供的ray batch")
        rays_o, rays_d = rays
        print(f"  rays_o device: {rays_o.device}, rays_d device: {rays_d.device}")

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            print(f"[render Debug] 使用静态相机生成光线")
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)    #每个像素的方向向量都被缩放成长度为 1 的单位向量。
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()
        print(f"[render Debug] viewdirs shape: {viewdirs.shape}, device: {viewdirs.device}")

    sh = rays_d.shape # [..., 3]
    print(f"[render Debug] 光线形状: {sh}")
    
    if ndc:
        # for forward facing scenes
        print(f"[render Debug] 应用NDC坐标转换")
        focal_val = K[0][0] if hasattr(K[0][0], 'item') else K[0][0]
        rays_o, rays_d = ndc_rays(H, W, focal_val, 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()
    print(f"[render Debug] 光线展平后:")
    print(f"  rays_o shape: {rays_o.shape}, device: {rays_o.device}")
    print(f"  rays_d shape: {rays_d.shape}, device: {rays_d.device}")

    # 确保near和far在正确的设备上
    rays_o_device = rays_o.device
    near_tensor = near * torch.ones_like(rays_d[...,:1])
    far_tensor = far * torch.ones_like(rays_d[...,:1])
    
    # 如果near/far是标量，确保它们在正确的设备上
    if isinstance(near, (int, float)):
        near_tensor = near_tensor.to(rays_o_device)
    if isinstance(far, (int, float)):
        far_tensor = far_tensor.to(rays_o_device)
        
    print(f"[render Debug] near_tensor device: {near_tensor.device}, far_tensor device: {far_tensor.device}")
    
    rays = torch.cat([rays_o, rays_d, near_tensor, far_tensor], -1)
    print(f"[render Debug] 光线batch创建完成: {rays.shape}, device: {rays.device}")
    
    if use_viewdirs:
        # 确保viewdirs在正确的设备上
        if viewdirs.device != rays.device:
            viewdirs = viewdirs.to(rays.device)
        rays = torch.cat([rays, viewdirs], -1)
        print(f"[render Debug] 添加viewdirs后: {rays.shape}, device: {rays.device}")

    # Render and reshape
    print(f"[render Debug] 开始批量渲染...")
    print('-'*40)
    all_ret = batchify_rays(rays, chunk, **kwargs)
    
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)
        print(f"[render Debug] 结果 {k}: shape {all_ret[k].shape}, device: {all_ret[k].device}")

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    
    print(f"[render Debug] === render 完成 ===\n")
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):
    print(f"[render_path Debug] === 开始 render_path ===")
    print(f"[render_path Debug] 输入参数:")
    print(f"  render_poses shape: {render_poses.shape}, device: {render_poses.device}")
    print(f"  hwf: {hwf}")
    print(f"  K shape: {K.shape}, K type: {type(K)}")
    print(f"  chunk: {chunk}")

    H, W, focal = hwf
    print(f"[render_path Debug] 解析hwf: H={H}, W={W}, focal={focal}")

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor
        print(f"[render_path Debug] 降采样后: H={H}, W={W}, focal={focal}")

    rgbs = []
    disps = []

    t = time.time()

    for i, c2w in enumerate(tqdm(render_poses)):
        print(f"\n[render_path Debug] 渲染第 {i} 个姿态")
        print(f"[render_path Debug] c2w shape: {c2w.shape}, device: {c2w.device}")
        print(f"  时间: {time.time() - t}")
        t = time.time()
        
        # 确保c2w在正确的设备上
        c2w = c2w.to(device)
        print(f"[render_path Debug] c2w移动到设备后: {c2w.device}")
        
        rgb, disp, acc, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        print(f"[render_path Debug] 渲染结果:")
        print(f"  rgb shape: {rgb.shape}, device: {rgb.device}")
        print(f"  disp shape: {disp.shape}, device: {disp.device}")
        
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        
        if i==0:
            print(f"[render_path Debug] 第一个结果的形状: rgb {rgb.shape}, disp {disp.shape}")

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    print(f"[render_path Debug] === render_path 完成 ===\n")
    return rgbs, disps


def create_nerf(args):
    """Instantiate NeRF's MLP model. 创建NeRF模型实例"""
    print(f"[create_nerf Debug] Creating NeRF model with multires: {args.multires}, use_viewdirs: {args.use_viewdirs}")
    
    # 获取位置编码器
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)
    print(f"[create_nerf Debug] Positional encoding input channels: {input_ch}")

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        # 获取视角方向编码器
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
        print(f"[create_nerf Debug] View direction encoding input channels: {input_ch_views}")
    
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    
    print(f"[create_nerf Debug] Network depth: {args.netdepth}, width: {args.netwidth}")
    print(f"[create_nerf Debug] Input channels: {input_ch}, view channels: {input_ch_views}, output channels: {output_ch}")
    
    # 创建模型
    print(f"[create_nerf Debug] Creating coarse network...")

    #-----------------------------------------------------
    print('-----------------------------------------------------')
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    grad_vars = list(model.parameters())

    print(f"[create_nerf Debug] Coarse network parameters: {sum(p.numel() for p in model.parameters())}")

    model_fine = None
    if args.N_importance > 0:
        print(f"[create_nerf Debug] Creating fine network...")
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        grad_vars += list(model_fine.parameters())
        print(f"[create_nerf Debug] Fine network parameters: {sum(p.numel() for p in model_fine.parameters())}")
        print(f"[create_nerf Debug] Total parameters: {sum(p.numel() for p in grad_vars)}")

    print('---------------------------------------------------------')

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)



    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))
    print(f"[create_nerf Debug] Optimizer created with learning rate: {args.lrate}")

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
        print(f"[create_nerf Debug] Using specified checkpoint path: {args.ft_path}")
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]
        print(f"[create_nerf Debug] Searching checkpoints in: {os.path.join(basedir, expname)}")

    print('[create_nerf Debug] Found ckpts', ckpts)

    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print(f'[create_nerf Debug] Reloading from: {ckpt_path}')
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step'] # 'global_step' - 训练步数/迭代次数
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])    # 'optimizer_state_dict' - 优化器的状态字典

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])    # 'network_fn_state_dict' - 粗网络模型的状态字典
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])  # 'network_fine_state_dict' - 精细网络模型的状态字典（如果有的话）
        print(f"[create_nerf Debug] Checkpoint loaded, global_step: {start}")
    else:
        print("[create_nerf Debug] No checkpoint found or reload disabled, starting from scratch")

    ##########################

    render_kwargs_train = {
    'network_query_fn' : network_query_fn,   # 用于在网络中查询点（MLP 的 forward 接口）
    'perturb' : args.perturb,                # 是否在采样时添加随机扰动（训练中常开）
    'N_importance' : args.N_importance,      # 精细网络采样点数量
    'network_fine' : model_fine,             # 精细网络模型（fine network）
    'N_samples' : args.N_samples,            # 粗网络采样点数量
    'network_fn' : model,                    # 粗网络模型（coarse network）
    'use_viewdirs' : args.use_viewdirs,      # 是否使用视角方向信息
    'white_bkgd' : args.white_bkgd,          # 是否使用白色背景
    'raw_noise_std' : args.raw_noise_std,    # 向输出密度添加的噪声（训练时有助于正则化）
}

    # NDC only good for LLFF-style forward facing data
    print(args.dataset_type, args.no_ndc)
    if args.dataset_type != 'llff' or args.no_ndc:
        
        print('[create_nerf Debug] Not using NDC coordinates')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp
    else:
        print('[create_nerf Debug] Using NDC coordinates')

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    print(f"[create_nerf Debug] Training render kwargs: perturb={render_kwargs_train['perturb']}, N_samples={render_kwargs_train['N_samples']}")
    print(f"[create_nerf Debug] Test render kwargs: perturb={render_kwargs_test['perturb']}, raw_noise_std={render_kwargs_test['raw_noise_std']}")

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def config_parser():
    """配置参数解析器"""
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')

    # training options 训练选项
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')

    # rendering options 渲染选项
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options 训练选项
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    # dataset options 数据集选项
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true',default=True, 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options 日志和保存选项
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000, 
                        help='frequency of render_poses video saving')

    return parser


def load_blender_data(basedir, half_res=False, testskip=1):
    pass

def load_LINEMOD_data(basedir, half_res=False, testskip=1): 
    pass

def load_dv_data(scene, basedir, testskip=1):
    pass

def train():
    # 主程序执行
    print("Initializing NeRF configuration...")
    parser = config_parser()
    args = parser.parse_args()

    # 打印关键参数
    print(f"\nKey NeRF Configuration:")
    print(f"  Experiment name: {args.expname}")
    print(f"  Dataset: {args.datadir}")
    print(f"  Network depth: {args.netdepth}, width: {args.netwidth}")
    print(f"  Positional encoding frequencies: {args.multires}")
    print(f"  Use view directions: {args.use_viewdirs}")
    print(f"  Coarse samples per ray: {args.N_samples}")
    print(f"  Fine samples per ray: {args.N_importance}")
    print(f"  Batch size: {args.N_rand}")
    print(f"  Learning rate: {args.lrate}")

    K = None
    print('args.dataset_type:', args.dataset_type)
    if args.dataset_type == 'llff':  #常用于神经场景表示任务，例如 NeRF
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)

        #llff提供了多分辨率图像（快慢结合），完整的几何信息：1.相机位姿：每张图像的3D位置和朝向；2.场景边界：整个场景的深度范围（near/far）；3.相机内参：焦距、图像尺寸等
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]   # 确保 i_test 是列表

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
            
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':   #常用于 3D 渲染任务
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'LINEMOD': #常用于物体检测、分割和姿态估计任务。它包含从不同视角拍摄的物体图像
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.half_res, args.testskip)
        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'deepvoxels':  #用于多视图立体任务，即从不同视角拍摄场景的多张图像

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])
        #相机内参矩阵（Camera Intrinsics Matrix），它定义了相机的内部参数，用于将3D相机坐标系中的点投影到2D图像平面上

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())


    print("\nCreating NeRF model...")
    # out = create_nerf(args)
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    print("NeRF model creation process completed.")

    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses
    # torch.Tensor(render_poses).to(device)
    print('-'*40)
    render_poses = torch.Tensor(render_poses).to(device)  # 添加赋值
    

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            rgbs, _ = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            return

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching
        print('get rays')
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        print('done, concats')
        rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
        rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)

        print('done')
        i_batch = 0

    # Move training data to GPU
    if use_batching:
        images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)


    N_iters = 200000 + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # Summary writers
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))
    
    start = start + 1
    for i in trange(start, N_iters):
        time0 = time.time()

        # Sample random ray batch
        if use_batching:
            # Random over all images
            batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0

        else:
            # Random from one image
            img_i = np.random.choice(i_train)
            target = images[img_i]
            target = torch.Tensor(target).to(device)
            pose = poses[img_i, :3,:4]

            if N_rand is not None:
                rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                if i < args.precrop_iters:
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                            torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                        ), -1)
                    if i == start:
                        print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        #####  Core optimization loop  #####
        rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)

        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)
        trans = extras['raw'][...,-1]
        loss = img_loss
        psnr = mse2psnr(img_loss)

        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        dt = time.time()-time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if i%args.i_video==0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

            # if args.use_viewdirs:
            #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
            #     with torch.no_grad():
            #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
            #     render_kwargs_test['c2w_staticcam'] = None
            #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

        if i%args.i_testset==0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')


    
        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
        """
            print(expname, i, psnr.numpy(), loss.numpy(), global_step.numpy())
            print('iter time {:.05f}'.format(dt))

            with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_print):
                tf.contrib.summary.scalar('loss', loss)
                tf.contrib.summary.scalar('psnr', psnr)
                tf.contrib.summary.histogram('tran', trans)
                if args.N_importance > 0:
                    tf.contrib.summary.scalar('psnr0', psnr0)


            if i%args.i_img==0:

                # Log a rendered validation view to Tensorboard
                img_i=np.random.choice(i_val)
                target = images[img_i]
                pose = poses[img_i, :3,:4]
                with torch.no_grad():
                    rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose,
                                                        **render_kwargs_test)

                psnr = mse2psnr(img2mse(rgb, target))

                with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):

                    tf.contrib.summary.image('rgb', to8b(rgb)[tf.newaxis])
                    tf.contrib.summary.image('disp', disp[tf.newaxis,...,tf.newaxis])
                    tf.contrib.summary.image('acc', acc[tf.newaxis,...,tf.newaxis])

                    tf.contrib.summary.scalar('psnr_holdout', psnr)
                    tf.contrib.summary.image('rgb_holdout', target[tf.newaxis])


                if args.N_importance > 0:

                    with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):
                        tf.contrib.summary.image('rgb0', to8b(extras['rgb0'])[tf.newaxis])
                        tf.contrib.summary.image('disp0', extras['disp0'][tf.newaxis,...,tf.newaxis])
                        tf.contrib.summary.image('z_std', extras['z_std'][tf.newaxis,...,tf.newaxis])
        """

        global_step += 1



    


train()