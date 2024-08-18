import trimesh
import numpy as np
import pickle
import cv2
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
import cv2
import torch
import numpy as np
from pytorch3d.renderer import (
    FoVOrthographicCameras,
    PerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    SoftPhongShader,
    PointLights
)
from pytorch3d.utils import cameras_from_opencv_projection
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import Textures
import hydra
# from pytorch3d.renderer.cameras import perspective_cameras_from_params
class VisMesh:
    def __init__(self, pids, info=None, vertices=None, objects=None):
        self.info = info
        self.pids = pids
        self.objects = objects  
        self.vertices = vertices
        with open (hydra.utils.to_absolute_path('./data/SMPLX/smpl/SMPL_NEUTRAL.pkl'), 'rb') as f:
            params = pickle.load(f, encoding='latin1')
        self.faces = params['f']
        self.meshes = self.create_obj_mesh()

    def create_obj_mesh(self):
        meshes = {}
        for pid in self.pids:
            if self.objects is not None:
                object = self.objects[pid]
                mesh = trimesh.Trimesh(vertices = object.feat['vertices'], faces = self.faces)
            elif self.vertices is not None:
                mesh = trimesh.Trimesh(vertices = self.vertices[pid], faces = self.faces)
            meshes[pid] = mesh
        return meshes

        
    def savemesh(self, pids=[], filename = 'debug/meshes.ply'):
        if pids == []:
            pids = self.pids
        mesh_list = []
        for pid in pids:
            mesh_list.append(self.meshes[pid])
        mesh_combined = trimesh.util.concatenate(mesh_list)
        
        trimesh.exchange.export.export_mesh(mesh_combined, filename, 'ply')

    def projection(self, cam_info, H, W, pid = '0', filename = 'debug/projection.jpg'):
        R = cam_info['R']
        T = cam_info['T']
        K = cam_info['K']

        vertice = self.vertices[pid]
        x_w = vertice.copy()
        x_c = x_w @ R.T + T.T
        x_p_homo = x_c @ K.T
        x_p = x_p_homo[:, :2]/x_p_homo[:, 2:]

        res = np.zeros((H, W, 3))
        pixel = x_p.round().astype(int)
        res[pixel[:, 1], pixel[:, 0]] = [1., 1., 1.]
        pred = (np.clip(res, 0, 1.) * 255).astype(np.uint8)
        pred = pred[..., [2, 1, 0]]
        cv2.imwrite(filename, pred)

        return x_p
    
    def projection_mask(self, cam_info, H, W, pid = '0', filename='debug/project_mask.png'):
        
        device = torch.device("cuda:0")
        R = torch.from_numpy(cam_info['R'].reshape(1, 3, 3)).to(torch.float32)
        T = torch.from_numpy(cam_info['T'].reshape(1, 3)).to(torch.float32)
        K = torch.from_numpy(cam_info['K'].reshape(1, 3, 3)).to(torch.float32)
        img_size = torch.tensor([[H, W]], device=device)
        cam = cameras_from_opencv_projection(R, T, K, img_size)       
        lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]],
                            ambient_color=((1,1,1),),diffuse_color=((0,0,0),),specular_color=((0,0,0),))
        camera = PerspectiveCameras(focal_length=cam.focal_length, principal_point=cam.principal_point,
                                    R = cam.R, T = cam.T, image_size = img_size, device=device)
        # camera = perspective_cameras_from_params(K = K.reshape(1, 3, 3), R = R, T = T, device=device)
        raster_settings = RasterizationSettings(image_size=(H, W),faces_per_pixel=1,blur_radius=0)
        rasterizer = MeshRasterizer(cameras=camera, raster_settings=raster_settings)
        shader = SoftPhongShader(cameras=camera, lights=lights, device=device)
        renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)
        verts = torch.from_numpy(self.vertices[pid].reshape(1, -1, 3)).to(device)
        faces = torch.from_numpy(self.faces.reshape(1, -1, 3).astype(np.int64)).to(device)
        # normals = torch.stack(mesh.verts_normals_list()) * 0.5 + 0.5
        mesh = Meshes(verts, faces, Textures(verts_rgb=torch.ones_like(verts)*0.3))
        img = renderer(meshes_world=mesh, cameras=camera)
        img = (img[0][:, :, :3].cpu().numpy()*255).astype(np.uint8)
        # cv2.imwrite('debug/tmp.jpg', img)
        mask = img[:, :, 0]<255
        cv2.imwrite(filename, mask.astype(np.uint8)*255)
        return mask
