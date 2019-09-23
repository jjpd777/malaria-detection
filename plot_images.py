import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import glob

malaria = Path("./cell_images/malaria")
non_malaria = Path("./cell_images/non-malaria")
malaria = malaria.glob("*.png")
non_malaria = non_malaria.glob("*.png")
samples = list(malaria) + list(non_malaria)


SMALL_SIZE = 12
BIG_SIZE = 20

plt.rc('axes', titlesize=BIG_SIZE)     # fontsize of the axes title
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels

f, ax = plt.subplots(2,5, figsize=(30,10))
for i in range(10):
    img = cv2.imread(str(samples[i]))
    ax[i//5, i%5].imshow(img)
    if i<5:
        ax[i//5, i%5].set_title("Malaria")
    else:
        ax[i//5, i%5].set_title("Normal")
    ax[i//5, i%5].axis()
    ax[i//5, i%5].set_aspect('auto')
plt.savefig("plotted_images.png")
