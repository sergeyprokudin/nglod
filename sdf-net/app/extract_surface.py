import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torch.nn as nn

import trimesh
import mcubes

from lib.models import *
from lib.options import parse_options


def extract_mesh(args):
    # Prepare directory
    ins_dir = os.path.join(args.mesh_dir, name)
    if not os.path.exists(ins_dir):
        os.makedirs(ins_dir)

    # Get SDFs
    with torch.no_grad():
        xx = torch.linspace(-1, 1, args.mc_resolution, device=device)
        pts = torch.stack(torch.meshgrid(xx, xx, xx), dim=-1).reshape(-1,3)
        chunks = torch.split(pts, args.batch_size)
        dists = []
        for chunk_pts in chunks:
            dists.append(net(chunk_pts).detach())

    # Convert to occupancy
    dists = torch.cat(dists, dim=0)
    grid = dists.reshape(args.mc_resolution, args.mc_resolution, args.mc_resolution)
    occupancy = torch.where(grid <= 0, 1, 0)

    # Meshify
    print('Fraction occupied: {:.5f}'.format((occupancy == 1).float().mean().item()))
    # vertices, triangles = mcubes.marching_cubes(occupancy.cpu().numpy(), 0.5) # Original post, small bug
    vertices, triangles = mcubes.marching_cubes(occupancy.cpu().numpy(), 0)

    # Resize + recenter
    b_min_np = np.array([-1., -1., -1.])
    b_max_np = np.array([ 1.,  1.,  1.])
    vertices = vertices / (args.mc_resolution - 1.0) * (b_max_np - b_min_np) + b_min_np

    # Save mesh
    mesh = trimesh.Trimesh(vertices, triangles)
    mesh_fname = os.path.join(ins_dir, f'mc_res{args.mc_resolution}.obj')
    print(f'Saving mesh to {mesh_fname}')
    mesh.export(mesh_fname)


if __name__ == '__main__':
    # Parse
    parser = parse_options(return_parser=True)
    app_group = parser.add_argument_group('app')
    app_group.add_argument('--mesh-dir', type=str, default='_results/render_app/meshes',
                           help='Directory to save the mesh')
    app_group.add_argument('--mc-resolution', type=int, default=256,
                           help='Marching cube grid resolution.')
    args = parser.parse_args()

    # Pick device
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # Get model
    if args.pretrained is not None:
        name = args.pretrained.split('/')[-1].split('.')[0]
    else:
        raise ValueError('No network weights specified!')
    net = globals()[args.net](args)
    net.load_state_dict(torch.load(args.pretrained), strict=False)
    net.to(device)
    net.eval()

    # Run Marching Cubes
    extract_mesh(args)
