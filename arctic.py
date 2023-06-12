import json
import torch
import numpy as np
import os.path as op
import render_utils
from PIL import Image
from glob import glob
from tqdm import tqdm
from loguru import logger
import traceback
import trimesh
import time
import neural_renderer as nr
import argparse

DEVICE = "cuda"


############################# load data ############################

with open("arctic_data/data/meta/misc.json", "r") as f:
    misc = json.load(f)

# unpack
subjects = list(misc.keys())
intris_mat = {}
ioi_offset = {}
for subject in subjects:
    intris_mat[subject] = misc[subject]["intris_mat"]
    ioi_offset[subject] = misc[subject]["ioi_offset"]

logger.info(f"load misc")
############################# face template ####################################

# MANO
sealed_faces = np.load("./data/meta_data/sealed_faces.npy", allow_pickle=True).item()
faces = sealed_faces["sealed_faces_right"]              # right hand (1554, 3) face vertices index,    1554 faces [0..778]
faces_color = sealed_faces["sealed_faces_color_right"]  # (1554,) 0,1,..15              16 faces color for one hand

mano_faces_r = torch.LongTensor(faces).to(DEVICE)
mano_faces_l = mano_faces_r[:, np.array([1, 0, 2])]  # opposite face normal
mano_faces_l += render_utils.N_VERTEX  # += 779 
mano_faces = torch.cat((mano_faces_r, mano_faces_l), dim=0) # ([3108, 3])

logger.info(f"load mano faces")
############################################################

def part_texture_object(OBJECT):
    mesh_o = trimesh.load(f'./arctic_data/data/meta/object_vtemplates/{OBJECT}/mesh.obj', process=False)
    obj_faces = mesh_o.faces   #(7950, 3)
    obj_faces = torch.LongTensor(obj_faces).to(DEVICE) 


    obj_faces += render_utils.N_VERTEX * 2
    all_faces =  torch.cat((mano_faces, obj_faces), dim=0)   #  torch.Size([ 11058, 3])
    all_faces = all_faces.unsqueeze(0)              # torch.Size([1, 11058, 3])

    # get_part_texture
    faces = all_faces[0].detach().cpu().numpy() # (11058, 3)
    face2label=faces_color

    num_faces = faces.shape[0]      # 11058
    half_faces = 1554

    face2label = face2label[:, None]
    # import pdb; pdb.set_trace()
    face2label = np.repeat(face2label, 3, axis=1) # 1554 x 3

    face_colors = np.ones((num_faces, 4)) # (11058, 4)
    face_colors[:half_faces, :3] = face2label
    face_colors[half_faces:2*half_faces, :3] = face2label + 16  #(3108, 4), 0..31

    H_, W_ = face_colors[2*half_faces:, :3].shape
    face_colors[2*half_faces:, :3] = np.full((H_, W_), 32)

    texture = np.zeros((1, faces.shape[0], 1, 1, 1, 3), dtype=np.float32)
    texture[0, :, 0, 0, 0, :] = face_colors[:, :3] / 33
    texture = torch.from_numpy(texture).float()      #torch.Size([1, 11058, 1, 1, 1, 3])

    part_texture = texture.to(DEVICE)           # [0, 0.969]
    return part_texture, all_faces
###############################################################################################

def process_segmentation(img_path, data, part_texture, all_faces):
    # print(img_path)  ./images/s01/capsulemachine_grab_01/0/00268.jpg
    
    sid, seq_name, view_idx, image_idx = img_path.split("/")[-4:]
    image_idx = int(image_idx.split(".")[0])
    vidx = image_idx - ioi_offset[sid]
    view_idx = int(view_idx)

    folder_path = op.dirname(
        img_path.replace("images",  "./outputs/segms")
    )
 
    # out folder and path
    render_utils.mkdir(folder_path)
    out_path = op.join(folder_path, op.basename(img_path).replace(".jpg", ".png"))
    
    #################################  vertices  ################################
    # import pdb; pdb.set_trace()
    # MANO mesh vertices, 778 single hand vertices
    v_r = data['cam_coord']['verts.right'][vidx][view_idx] * 1000    # (732, 10, 778, 3) -> (778, 3)
    v_l = data['cam_coord']['verts.left' ][vidx][view_idx] * 1000 
    v_r = render_utils.add_seal_vertex(v_r) # (779, 3)
    v_l = render_utils.add_seal_vertex(v_l)
    v_r = v_r.reshape(1, -1, 3)     # (1, 779, 3)
    v_l = v_l.reshape(1, -1, 3) 

    # Object mesh vertices   
    v_o = data['cam_coord']['verts.object'][vidx][view_idx] * 1000   # (3947, 3)
    v_o = v_o.reshape(1, -1, 3)   #(1, 3947, 3)

    ############################## get intrinsic K ################################
    
    if view_idx == 0:
        intrx = data['params']["K_ego"][vidx].copy()
    else:
        intrx = np.array(intris_mat[sid][view_idx - 1])
    
    # K = intrx

    focal= np.array([intrx[0][0], intrx[1][1]])
    princpt= np.array([intrx[0][2], intrx[1][2]])

    ################################## render color ####################################
    # im = Image.open(img_path)
    # print(im.size)
    im_size = (2000, 2800) 
    im_w, im_h = im_size

    imsize = max(im_size) + 10

    # initialize neural renderer
    neural_renderer = nr.Renderer(
        dist_coeffs=None,
        orig_size=imsize,
        image_size=imsize,
        light_intensity_ambient=1,
        light_intensity_directional=0,
        anti_aliasing=False,
    ).cuda()

    scale = 1.0
    K = torch.FloatTensor(
        np.array(
            [[[focal[0], scale, princpt[0]], [0, focal[1], princpt[1]], [0, 0, 1]]]
        )
    ).to(DEVICE)

    N_PARTS = 16 * 2 + 1

    bins = (torch.arange(int(N_PARTS)) / float(N_PARTS) * 255.0) + 1
    bins = bins.to(DEVICE)

    # MANO is rotated 180 degrees in x axis, revert it.
    vertices_l = torch.FloatTensor(v_l).to(DEVICE) / 1000
    vertices_r = torch.FloatTensor(v_r).to(DEVICE) / 1000
    vertices_o = torch.FloatTensor(v_o).to(DEVICE) / 1000

    vertices = torch.cat((vertices_r, vertices_l, vertices_o), dim=1)

    R = torch.eye(3).to(DEVICE)
    cam_t = torch.zeros(1, 3).to(DEVICE)

    parts, render, depth = render_utils.generate_part_labels(
        vertices=vertices,
        faces=all_faces,
        cam_t=cam_t,
        K=K,
        R=R,
        part_texture=part_texture,
        neural_renderer=neural_renderer,
        part_bins=bins,
    )

    # below is needed for visualization only
    parts = parts.cpu().numpy()
    ######################## output ######################

    parts = parts[0].astype(np.uint8)
    parts = parts[:im_h, :im_w] #(2800, 2000)  [0, 1,..33, ..]

    parts_im = Image.fromarray(parts.astype(np.uint8))
    parts_im.save(out_path)

    # im_arr = np.array(Image.open(out_path), dtype=np.uint8)
    # assert np.abs(im_arr - parts).sum() == 0



# def main2():
#     seq = glob(f"./arctic_data/data/images/*/*")
#     pbar = tqdm(seq)
#     for seq in pbar: 
#         # seq = 'images/s09/waffleiron_use_03'
#         sid, seq_name = seq.split("/")[1:]
#         if sid == 's03':
#             continue
        
#         data = np.load(f'./outputs/processed_verts/seqs/{sid}/{seq_name}.npy', allow_pickle=True).item()
#         # dict_keys(['world_coord', 'cam_coord', '2d', 'bbox', 'params'])
#         OBEJCT = seq_name.split('_')[0]
#         part_texture, all_faces = part_texture_object(OBEJCT)

#         frames =  glob(f"{seq}/*/*.jpg")
#         sbar = tqdm(frames)

#         for img_path in sbar:
#             sbar.set_description("Processing %s" % img_path)
#             try:
#                 process_segmentation(img_path, data, part_texture, all_faces)
#             except Exception as e:
#                 logger.info(traceback.format_exc())
#                 time.sleep(2)
#                 logger.info(f"Failed at {img_path}")


def construct_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=int, default=None)
    args = parser.parse_args()
    return args


def main(args):
    with open(
        f"./arctic_data/data/splits_json/protocol_p1.json", "r"
    ) as f:
        prot = json.load(f)

    # train : 0..266 prot['train'][266]
    # val   : 0..33  prot['val'][33]
    if args.task_id < 0:
        logger.info(f"please give task_id, expect number from 0 to 300")
        return 
    if args.task_id < 267:
        seq = prot['train'][args.task_id]
    elif args.task_id < 301:
        seq = prot['val'][args.task_id - 267]
    else:
        logger.info(f"invalid number {args.task_id}, expect number from 0 to 300")
        return 

    # seq = 's01/espressomachine_use_02'
    seq = f"images/{seq}" # path to full images seq 'images/s01/espressomachine_use_02'

    sid, seq_name = seq.split("/")[1:]
    
    data = np.load(f'./outputs/processed_verts/seqs/{sid}/{seq_name}.npy', allow_pickle=True).item()
    # dict_keys(['world_coord', 'cam_coord', '2d', 'bbox', 'params'])
    
    OBEJCT = seq_name.split('_')[0]
    part_texture, all_faces = part_texture_object(OBEJCT)

    frames =  glob(f"{seq}/*/*.jpg")
    sbar = tqdm(frames)

    for img_path in sbar:
        sbar.set_description("Processing %s" % img_path)
        try:
            process_segmentation(img_path, data, part_texture, all_faces)
        except Exception as e:
            logger.info(traceback.format_exc())
            time.sleep(2)
            logger.info(f"Failed at {img_path}")

if __name__ == "__main__":
    args = construct_args()
    main(args)