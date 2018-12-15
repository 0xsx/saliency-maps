#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Loads a given RGB image and computes an Itti-style saliency map. For details,
see: Itti, Laurent, Christof Koch, and Ernst Niebur. "A model of saliency-based
visual attention for rapid scene analysis." IEEE Transactions on pattern
analysis and machine intelligence. 1998.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import cv2
import numpy as np

from os import path




FLOAT_DTYPE = "float32"
_EPSILON = float(1e-12)





def normalize_array_range(arr):
  """Returns a new array with `arr` values scaled within the range [0.0, 1.0]."""
  
  min_val = np.min(arr)
  max_val = np.max(arr)

  scaled = np.copy(arr)

  if min_val == max_val:
    if max_val >= 1:
      return np.ones_like(arr)
    elif max_val <= 0:
      return np.zeros_like(arr)
    else:
      return scaled

  if max_val < 0:
    scaled -= max_val
    min_val -= max_val
    max_val = 0.

  if min_val < 0:
    scaled -= min_val
    max_val -= min_val
    min_val = 0.

  if max_val:
    scaled /= max_val

  return scaled








def read_rgb_image(in_file):
  """Reads the specified image file as a numpy float array normalized
  between [0.0, 1.0] and returns a RGB image of shape (height, width, 3)
  containing red, green, and blue channels on the last dimension."""

  img = cv2.imread(in_file, cv2.IMREAD_COLOR).astype(FLOAT_DTYPE)

  img = normalize_array_range(img)

  b_channel = img[:, :, 0]
  g_channel = img[:, :, 1]
  r_channel = img[:, :, 2]

  img = np.zeros((r_channel.shape[0], r_channel.shape[1], 3), dtype=FLOAT_DTYPE)
  img[:, :, 0] = r_channel
  img[:, :, 1] = g_channel
  img[:, :, 2] = b_channel


  return img





def compute_saliency_map(rgb_img):
  """Computes Itti-style visual saliency maps for `rgb_img`. Returns a saliency
  map array of shape `(height, width)`."""


  def __resize(img, shape, nearest=False):
    return cv2.resize(img, shape[::-1])

  def __gabor_filter(img, freq, theta, sigma=0.1):
    kernel = cv2.getGaborKernel((6, 6), sigma, theta, 1./freq, 1, ktype=cv2.CV_32F)
    return cv2.filter2D(img, cv2.CV_32F, kernel)



  feature_maps = []
  layer_maps = []

  r_channel = rgb_img[:, :, 0]
  g_channel = rgb_img[:, :, 1]
  b_channel = rgb_img[:, :, 2]

  for layer in range(7):
    r_channel = cv2.pyrDown(r_channel)
    g_channel = cv2.pyrDown(g_channel)
    b_channel = cv2.pyrDown(b_channel)

    if layer < 1:
      continue

    intensity_map = (r_channel + g_channel + b_channel) / 3.

    gabor_maps = [__gabor_filter(intensity_map, 1.0, i / 4. * np.pi, 3) for i in range(4)]

    nr_map = g_channel - b_channel
    ng_map = r_channel - (r_channel + g_channel) / 2.
    nb_map = b_channel - (r_channel + g_channel) / 2.
    ny_map = (r_channel + g_channel) / 2. - np.fabs(r_channel - g_channel) - b_channel

    nr_map[nr_map < 0.] = 0.
    ng_map[ng_map < 0.] = 0.
    nb_map[nb_map < 0.] = 0.
    ny_map[ny_map < 0.] = 0.

    intensity_mask = intensity_map < 0.5 * intensity_map.mean()
    nr_map[intensity_mask] = 0
    ng_map[intensity_mask] = 0
    nb_map[intensity_mask] = 0
    ny_map[intensity_mask] = 0


    layer_maps.append((intensity_map, nr_map, ng_map, nb_map, ny_map, gabor_maps))


    # compute layer difference features

    if layer == 3:
      top_inds = [0]
    elif layer == 4:
      top_inds = [0, 1]
    elif layer == 5:
      top_inds = [1, 2]
    elif layer == 6:
      top_inds = [2]
    else:
      top_inds = []

    for ind in top_inds:
      intensity_map_top, nr_map_top, ng_map_top, nb_map_top, ny_map_top, gabor_maps_top = layer_maps[ind]

      intensity_feats = np.fabs(intensity_map_top - __resize(intensity_map, intensity_map_top.shape))

      gabor_feats = [np.fabs(gabor_maps_top[i] - __resize(gabor_maps[i], gabor_maps_top[i].shape))
                     for i in range(4)]

      bg_feats = np.fabs((nb_map_top - ng_map_top) - __resize(ng_map - nb_map, nb_map_top.shape))
      ry_feats = np.fabs((nr_map_top - ny_map_top) - __resize(ny_map - nr_map, nb_map_top.shape))

      feature_maps.append((intensity_feats, bg_feats, ry_feats, gabor_feats))



  # make conspicuity maps
  sal_map = np.zeros((rgb_img.shape[0], rgb_img.shape[1]), dtype=FLOAT_DTYPE)

  intensity_consp = np.zeros_like(sal_map)
  color_consp = np.zeros_like(sal_map)
  gabor_consp = np.zeros_like(sal_map)

  for intensity_feats, bg_feats, ry_feats, gabor_feats in feature_maps:
    intensity_consp += __resize(normalize_array_range(intensity_feats), sal_map.shape)
    color_consp += __resize(normalize_array_range(bg_feats), sal_map.shape)
    color_consp += __resize(normalize_array_range(ry_feats), sal_map.shape)
    for gf in gabor_feats:
      gabor_consp += __resize(normalize_array_range(gf), sal_map.shape)

  sal_map += normalize_array_range(intensity_consp) + normalize_array_range(color_consp) + normalize_array_range(gabor_consp)
  sal_map /= 3.

  sal_map = cv2.GaussianBlur(sal_map, (7, 7), 10)


  return sal_map
  



def save_image(img, out_file):
  """Writes the specified image to the output file."""

  img = normalize_array_range(img)

  img = img * 255
  img[img > 255] = 255
  img[img < 0] = 0
  
  cv2.imwrite(out_file, img)







def main(in_file, out_file):
  """Main function called with command-line args."""

  rgb_img = read_rgb_image(in_file)
  sal_map = compute_saliency_map(rgb_img)
  save_image(sal_map, out_file)






if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(description=__doc__)

  parser.add_argument("in_file", help="Input image")
  parser.add_argument("out_file", help="Output saliency map")


  args = parser.parse_args()
  main(path.abspath(args.in_file), path.abspath(args.out_file))

