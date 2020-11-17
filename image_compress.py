#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import time

from PIL import Image

fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(16,10))

img = Image.open("img/miami.jpg")
axs[0, 0].imshow(img)
axs[0, 0].set_title("size = {}".format(img.size), size=16)
axs[0, 0].axis('off')

# convert it to gray pic
img_gray = img.convert('LA')

# convert it to np array
img_mat = np.array(list(img_gray.getdata(band=0)), float)
# reshape the np array
img_mat.shape = (img_gray.size[1], img_gray.size[0])
img_mat = np.matrix(img_mat)
axs[0, 1].imshow(img_mat, cmap='gray')
axs[0, 1].set_title('Gray', size=16)
axs[0, 1].axis('off')

# svd decomposition
U, S, V = np.linalg.svd(img_mat)

# Computing an approximation of the image using the first column of U and first row of V reproduces the most prominent feature of the image. Each column of pixels in this image is a different weighting of the same values, u_1.
for k,ax in zip([1, 10, 50, 100], axs.flatten()[2:]):
    img_recons = np.matrix(U[:, :k]) * np.diag(S[:k]) * np.matrix(V[:k, :])
    ax.imshow(img_recons, cmap='gray')
    title = "k = %s" % k
    ax.set_title(title, size=16)
    ax.axis('off')

fig.subplots_adjust(hspace=0)

plt.tight_layout()
plt.savefig('reconstructed_images_using_different_SVs.jpg', dpi=300)
