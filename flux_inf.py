from .resampler import Resampler
import os
from comfy import model_management
import torch
from insightface.app import FaceAnalysis
from insightface.utils import face_align
import folder_paths
import zipfile
import cv2
from insightface.utils.download import download_file
from insightface.utils.storage import BASE_REPO_URL
import numpy as np
from facexlib.recognition import init_recognition_model


# from ComfyUI_PuLID_Flux_ll.pulidflux import set_extra_config_model_path
def set_extra_config_model_path(extra_config_models_dir_key, models_dir_name:str):
    models_dir_default = os.path.join(folder_paths.models_dir, models_dir_name)
    if extra_config_models_dir_key not in folder_paths.folder_names_and_paths:
        folder_paths.folder_names_and_paths[extra_config_models_dir_key] = (
            [os.path.join(folder_paths.models_dir, models_dir_name)], folder_paths.supported_pt_extensions)
    else:
        if not os.path.exists(models_dir_default):
            os.makedirs(models_dir_default, exist_ok=True)
        folder_paths.add_model_folder_path(extra_config_models_dir_key, models_dir_default, is_default=True)

set_extra_config_model_path("insightface", "insightface")
set_extra_config_model_path("facexlib", "facexlib")
set_extra_config_model_path("inf-you", "inf-you")

INSIGHTFACE_DIR = folder_paths.get_folder_paths("insightface")[0]
FACEXLIB_DIR = folder_paths.get_folder_paths("facexlib")[0]
INFUSE_YOU_DIR = folder_paths.get_folder_paths("inf-you")[0]

def load_image_proj_model(proj_model_name, device, dtype, image_proj_num_tokens=8):
        # Load image proj model
        num_tokens = image_proj_num_tokens
        image_emb_dim = 512
        image_proj_model = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=num_tokens,
            embedding_dim=image_emb_dim,
            output_dim=4096,
            ff_mult=4,
        )
        image_proj_model_path = os.path.join(INFUSE_YOU_DIR, proj_model_name)
        ipm_state_dict = torch.load(image_proj_model_path, map_location="cpu")
        image_proj_model.load_state_dict(ipm_state_dict['image_proj'])
        del ipm_state_dict
        image_proj_model.to(device, dtype=dtype)
        image_proj_model.eval()

        return image_proj_model

def download_insightface_model(sub_dir, name, force=False, root='~/.insightface'):
    # Copied and modified from insightface.utils.storage.download
    # Solve https://github.com/deepinsight/insightface/issues/2711
    _root = os.path.expanduser(root)
    dir_path = os.path.join(_root, sub_dir, name)
    if os.path.exists(dir_path) and not force:
        return dir_path
    zip_file_path = os.path.join(_root, sub_dir, name + '.zip')
    model_url = "%s/%s.zip"%(BASE_REPO_URL, name)
    download_file(model_url,
             path=zip_file_path,
             overwrite=True)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # zip file has contains ${name}
    real_dir_path = os.path.join(_root, sub_dir)
    with zipfile.ZipFile(zip_file_path) as zf:
        zf.extractall(real_dir_path)
    #os.remove(zip_file_path)
    return dir_path

def tensor_to_image(tensor):
    if tensor.ndim == 4:
        tensor = tensor[0]
    image = tensor.mul(255).clamp(0, 255).byte().cpu()
    image = image[..., [2, 1, 0]].numpy()
    return image

def image_to_tensor(image):
    # Convert numpy array (H, W, C) to tensor (B, C, H, W)
    image = torch.from_numpy(image).float()
    # Move channels axis from last to second position
    image = image.permute(0, 1, 2)
    # Add batch dimension
    image = image.unsqueeze(0)
    # Normalize to [0, 1] range
    image = image / 255.0
    # Ensure values are clamped between 0 and 1
    image = image.clamp(0, 1)
    return image

class InfiniteYouInsightFaceLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "provider": (["CPU", "CUDA", "ROCM"], ),
            },
        }

    RETURN_TYPES = ("FACEANALYSIS",)
    FUNCTION = "load_insightface"
    CATEGORY = "infuse_you"

    def load_insightface(self, provider):
        name = "antelopev2"
        download_insightface_model("models", name, root=INSIGHTFACE_DIR)
        model = FaceAnalysis(name=name, root=INSIGHTFACE_DIR, providers=[provider + 'ExecutionProvider', ]) # alternative to buffalo_l

        return (model,)

class InfiniteYouApply:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                             "negative": ("CONDITIONING", ),
                             "infuse_net": ("CONTROL_NET", ),
                             "face_analysis": ("FACEANALYSIS", ),
                             "ref_image": ("IMAGE", ),
                             "proj_model_name": (folder_paths.get_filename_list("inf-you"),),
                             "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                             "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                             "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                             "vae": ("VAE", )
                             },
                "optional": {
                             "control_image": ("IMAGE", )
                             }
    }

    RETURN_TYPES = ("CONDITIONING","CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "apply_infiniteYou"

    CATEGORY = "conditioning/controlnet"

    def init_face_analysis(self, face_analysis):
                # Load face encoder with multiple detection sizes
        self.app_640 = face_analysis
        self.app_640.prepare(ctx_id=0, det_size=(640, 640))

        self.app_320 = face_analysis
        self.app_320.prepare(ctx_id=0, det_size=(320, 320))

        self.app_160 = face_analysis
        self.app_160.prepare(ctx_id=0, det_size=(160, 160))

        self.arcface_model = init_recognition_model('arcface', device='cuda')

    def _detect_face(self, id_image_cv2):
        # Try with largest detection size first
        face_info = self.app_640.get(id_image_cv2)
        if len(face_info) > 0:
            return face_info
        
        # If no face found, try medium detection size
        face_info = self.app_320.get(id_image_cv2)
        if len(face_info) > 0:
            return face_info

        # If still no face found, try smallest detection size
        face_info = self.app_160.get(id_image_cv2)
        return face_info

    def extract_arcface_bgr_embedding(self, in_image, landmark, arcface_model=None, in_settings=None):
        kps = landmark
        arc_face_image = face_align.norm_crop(in_image, landmark=np.array(kps), image_size=112)
        arc_face_image = torch.from_numpy(arc_face_image).unsqueeze(0).permute(0,3,1,2) / 255.
        arc_face_image = 2 * arc_face_image - 1
        arc_face_image = arc_face_image.cuda().contiguous()
        if arcface_model is None:
            arcface_model = self.arcface_model
        face_emb = arcface_model(arc_face_image)[0] # [512], normalized
        return face_emb

    def get_face_embedding(self, ref_image, device, dtype):
        # convert BRGB to BBGR
        ref_image = tensor_to_image(ref_image)
        self.image_proj_model = load_image_proj_model(self.image_proj_model, device, dtype)

        face_info = self._detect_face(ref_image)
        if len(face_info) == 0:
            raise Warning('No face detected in the input ID image')
        
        face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1] # only use the maximum face
        landmark = face_info['kps']
        id_embed = self.extract_arcface_bgr_embedding(ref_image, landmark)
        id_embed = id_embed.clone().unsqueeze(0).float().cuda()
        id_embed = id_embed.reshape([1, -1, 512])
        id_embed = id_embed.to(device=device, dtype=dtype)
        with torch.no_grad():
            id_embed = self.image_proj_model(id_embed)
            bs_embed, seq_len, _ = id_embed.shape
            id_embed = id_embed.repeat(1, 1, 1)
            id_embed = id_embed.view(bs_embed * 1, seq_len, -1)
            id_embed = id_embed.to(device=device, dtype=dtype)
            return id_embed
    
        return None
        
        

    def apply_infiniteYou(self, positive, negative, infuse_net, ref_image, proj_model_name, face_analysis, strength, start_percent, end_percent, vae=None, extra_concat=[], mask=None, control_image=None):

        device = model_management.get_torch_device()
        dtype = model_management.unet_dtype()

        self.init_face_analysis(face_analysis)

        self.image_proj_model = proj_model_name

        id_embed = self.get_face_embedding(ref_image, device, dtype)
        # control image is a tensor of zeros with the same 8 times width and height as the latent image
        if control_image is None:
            control_image = torch.zeros((1, 512, 512, 3), dtype=dtype, device=device)
        
        if strength == 0:
            return (positive, negative)

        control_hint = control_image.movedim(-1,1)
        cnets = {}

        out = []
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                d = t[1].copy()

                prev_cnet = d.get('control', None)
                if prev_cnet in cnets:
                    c_net = cnets[prev_cnet]
                else:
                    c_net = infuse_net.copy().set_cond_hint(control_hint, strength, (start_percent, end_percent), vae=vae, extra_concat=extra_concat)
                    c_net.set_previous_controlnet(prev_cnet)
                    cnets[prev_cnet] = c_net

                d['control'] = c_net
                d['control_apply_to_uncond'] = False
                if id_embed is not None:
                    d['cross_attn_controlnet'] = id_embed
                n = [t[0], d]
                c.append(n)
            out.append(c)
        return (out[0], out[1])


class InfiniteYouControlImagePreprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "face_analysis": ("FACEANALYSIS",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "preprocess_control_image"

    CATEGORY = "infuse_you"

    def init_face_analysis(self, face_analysis):
        # Load face encoder with multiple detection sizes
        self.app_640 = face_analysis
        self.app_640.prepare(ctx_id=0, det_size=(640, 640))

        self.app_320 = face_analysis
        self.app_320.prepare(ctx_id=0, det_size=(320, 320))

        # self.app_160 = faceanalysis
        # self.app_160.prepare(ctx_id=0, det_size=(160, 160))

        self.arcface_model = init_recognition_model('arcface', device='cuda')

    def _detect_face(self, id_image_cv2):
        # Try with largest detection size first
        face_info = self.app_640.get(id_image_cv2)
        if len(face_info) > 0:
            return face_info
        
        # If no face found, try medium detection size
        face_info = self.app_320.get(id_image_cv2)
        if len(face_info) > 0:
            return face_info

        # # If still no face found, try smallest detection size
        # face_info = self.app_160.get(id_image_cv2)
        return face_info

    def draw_kps(self, image_pil, kps, color_list=[(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]):
        stickwidth = 4
        limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
        kps = np.array(kps)
        
        w, h = image_pil.shape[1], image_pil.shape[0]
        out_img = np.zeros([h, w, 3])

        for i in range(len(limbSeq)):
            index = limbSeq[i]
            color = color_list[index[0]]

            x = kps[index][:, 0]
            y = kps[index][:, 1]
            length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
            angle = np.degrees(np.arctan2(y[0] - y[1], x[0] - x[1]))
            polygon = cv2.ellipse2Poly((int(np.mean(x)), int(np.mean(y))), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
        out_img = (out_img * 0.6).astype(np.uint8)

        for idx_kp, kp in enumerate(kps):
            color = color_list[idx_kp]
            x, y = kp
            out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

        return out_img

    def preprocess_control_image(self, image, face_analysis):
        # tensor to image
        image = tensor_to_image(image)
        self.init_face_analysis(face_analysis)
        face_info = self._detect_face(image)
        face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1] # only use the maximum face
        if len(face_info) == 0:
            raise Warning('No face detected in the input ID image')
        landmark = face_info['kps']
        out_img = self.draw_kps(image, landmark)
        out_img = image_to_tensor(out_img)
        return (out_img, )


NODE_CLASS_MAPPINGS = {
    "InfiniteYouApply": InfiniteYouApply,
    "InfiniteYouInsightFaceLoader": InfiniteYouInsightFaceLoader,
    "InfiniteYouControlImagePreprocessor": InfiniteYouControlImagePreprocessor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InfiniteYouApply": "Infinite You Apply",
    "InfiniteYouInsightFaceLoader": "Infinite You Insight Face Loader",
    "InfiniteYouControlImagePreprocessor": "Infinite You Control Image Preprocessor",
}
