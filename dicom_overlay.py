#!/usr/bin/env python
# coding: utf-8
#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D
import os,time,subprocess,glob,re
import pandas
import ipywidgets as ipyw
import pydicom as dicom
import argparse,sys

## please change here
def weight_img_two(im2,s0=-200,s1=200,r=30): # well-shaped potential
    wl = 1.0-0.5/r * (im2 - (s0-r)) # 1 at s0-r and 0 at s0+r
    wr = 0.5/r * (im2 - (s1-r))
    w = np.clip(np.maximum(wl,wr),0,1)
    return((w,1-w))

def weight_img_three(im,s0=-200,s1=200,r=30):
    w0 = np.clip(1.0-0.5/r * (im - (s0-r)), 0, 1) # 1 at s0-r and 0 at s0+r
    w2 = np.clip(0.5/r * (im - (s1-r)), 0, 1) # 0 at s1-r and 1 at s1+r
    return((w0,1-w0-w2,w2))

if __name__== "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument('--output', '-o', default='output', help="output dir")
    parser.add_argument('--in1', '-i1', default='full', help="low or full range input dir")
    parser.add_argument('--in2', '-i2', default='partial', help="middle range input dir")
    parser.add_argument('--in3', '-i3', default=None, help="high range input dir")
    parser.add_argument('--org', '-i0', default='org', help="original image dir")
    parser.add_argument('--truth', '-t', default=None, help="ground truth image dir (for evaluation)")
    parser.add_argument('--lower', default=-200, help="lower value of the transition range")
    parser.add_argument('--upper', default=+200, help="upper value of the transition range")
    parser.add_argument('--transition_margin', '-tm', default=30, help="margin of the transition")
    parser.add_argument('--clip', '-c', default=[-1100,1500], nargs=2, help="clipping bounds")
    parser.add_argument('--plot', '-p', action="store_true")

    args = parser.parse_args()

    # abspath = os.path.abspath(__file__)
    # args.root = os.path.join(os.path.dirname(abspath),"overlay")
    # args.in1 = os.path.join(args.root,"full")
    # args.in2 = os.path.join(args.root,"narrow")
    # args.org = os.path.join(args.root,"org")
    # args.truth = os.path.join(args.root,"truth")

    if args.plot:
        print(args)
    os.makedirs(args.output, exist_ok=True)
    input_dirs = [dn for dn in [args.in1,args.in2,args.in3] if dn is not None]

    m, M = args.clip
    rM=100

    dirlist = ["."]
    for f in os.listdir(args.org):
        if os.path.isdir(os.path.join(args.org,f)):
            dirlist.append(f)
    for dirname in dirlist:
        print("Processing... {}".format(dirname))
        fns = [os.path.basename(f) for f in glob.glob(os.path.join(args.org,dirname,'*.[dD][cC][mM]'))]
        if len(fns)==0:
            continue
        os.makedirs(os.path.join(args.output,dirname), exist_ok=True)
        for fn in fns:
            dcm_in = [dicom.read_file(os.path.join(fd,dirname,fn), force=True) for fd in input_dirs]
            dcm_org = dicom.read_file(os.path.join(args.org,dirname,fn), force=True)
            img_in = [np.clip(dcm_in[i].pixel_array.astype(np.float64) + dcm_in[i].RescaleIntercept, m,M) for i in range(len(dcm_in))]
            img_org = dcm_org.pixel_array.astype(np.float64) + dcm_org.RescaleIntercept
            if len(input_dirs) == 3:
                weight = weight_img_three(img_in[1], args.lower, args.upper, args.transition_margin)
            else:
                weight = weight_img_two(img_in[1], args.lower, args.upper, args.transition_margin)
            img_ov = sum([weight[i]*img_in[i] for i in range(len(weight))])
            img_ov = np.where(dcm_org.pixel_array + dcm_org.RescaleIntercept == -2048, -2048, img_ov)
            # save dicom
            img = (img_ov - dcm_org.RescaleIntercept).astype(dcm_org.pixel_array.dtype)   
            #print("{}: min HU {}, max HU {}".format(fn, np.min(img),np.max(img)))
        #            print(img.shape, img.dtype)
            dcm_org.PixelData = img.tobytes()
            dcm_org.save_as(os.path.join(args.output,dirname,fn))
            if args.plot:
                dcm_truth = dicom.read_file(os.path.join(args.truth,dirname,fn), force=True)
                img_truth = np.clip(dcm_truth.pixel_array.astype(np.float64) + dcm_truth.RescaleIntercept, m,M)
                # figure
                fig = plt.figure(figsize=(15,10))
                # for i,im in enumerate(imgs):
                #     ax = fig.add_subplot(1,len(imgs),i+1)
                #     ax.imshow(img_in1, cmap="coolwarm", vmin=m, vmax=M)
                # plt.show()
                #print(img_truth.min(),img_truth.max(), img_truth[50,256])
                img_ov = np.clip(img_ov,m,M)
                diff_ov = (img_ov-img_truth)[img_org>-990]  ## body only
                diff_in1 = (img_in[0]-img_truth)[img_org>-990]  ## body only
                plt.hist([diff_ov.ravel(),diff_in1.ravel()],bins=100,range=[-rM,rM],density=True,label=["overlay","in1"])
                plt.legend(loc='upper right')
                pc_ov = np.percentile(np.abs(diff_ov).ravel(), [5,95])
                pc_in1 = np.percentile(np.abs(diff_in1).ravel(), [5,95])
                plt.title("overlay -- mean abs err: {}, median err: {}, 5 and 95 percentile {},{}\nIn1  -- mean abs err: {}, median err: {}, 5 and 95 percentile {},{}".format(np.abs(diff_ov).mean(),np.median(diff_ov),pc_ov[0],pc_ov[1],np.abs(diff_in1).mean(),np.median(diff_in1),pc_in1[0],pc_in1[1]))
                plt.savefig(os.path.join(args.output,dirname,"error_{}.jpg".format(fn)))
                # 
                fig = plt.figure(figsize=(21,10))
                axes = fig.subplots(1, 2)
                im=axes[0].imshow(img_ov-img_truth, cmap="coolwarm", vmin=-50, vmax=50)
                axes[0].set_title("overlay")
                im=axes[1].imshow(img_in[0]-img_truth, cmap="coolwarm", vmin=-50, vmax=50)
                axes[1].set_title("In1")
                fig.subplots_adjust(right=0.8)
                cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
                fig.colorbar(im, cax=cbar_ax)
                plt.savefig(os.path.join(args.output,dirname,"diff_{}.jpg".format(fn)))
                #
                fig = plt.figure(figsize=(21,10))
                axes = fig.subplots(1, 1)
                axes.hist([img_ov.ravel(),img_in[0].ravel(),img_truth.ravel()],range=(m,M),bins=50, label=["overlay","in1","tr"])
                axes.legend(loc='upper right')
                # axes[1].hist([img_in1.ravel(),img_in2.ravel()],range=(m,M),bins=50, label=["in1","in2"])
                # axes[1].legend(loc='upper right')
                #axes[1].hist([img_org.ravel(),img_truth.ravel()],range=(m,M),bins=50, label=["org","tr"])
                #axes[1].legend(loc='upper right')
                plt.savefig(os.path.join(args.output,dirname,"hist_{}.jpg".format(fn)))
    print("Results are found in '{}'".format(args.output))

# %%
