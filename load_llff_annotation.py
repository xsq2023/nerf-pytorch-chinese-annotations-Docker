import numpy as np
import os, imageio


# 加载 poses_bounds.npy 文件
poses_bounds = np.load('/root/work/nerf-pytorch/data/nerf_llff_data/fern/poses_bounds.npy')

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
    print(f"c2w after concat shape: {c2w.shape}")
    
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])# tile--》在各个维度上可重复扩展bo
    print(f"bottom after tile shape: {bottom.shape}")
    
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)
    print(f"poses after concat shape: {poses.shape}")

    poses = np.linalg.inv(c2w) @ poses
    print(f"poses after inv @ shape: {poses.shape}")
    
    poses_[:,:3,:4] = poses[:,:3,:4]
    print(f"poses_ after assignment shape: {poses.shape}")
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



def load_llff_data(basedir, factor=8, recenter=True, bd_factor=.75, spherify=False, path_zflat=False):
    
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


load_llff_data('/root/work/nerf-pytorch/data/nerf_llff_data/fern', factor=8, recenter=True, bd_factor=.75, spherify=True, path_zflat=False)
