"""Live display based on https://stackoverflow.com/questions/42867400/python-show-image-upon-hovering-over-a-point"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np


def scale0to1(img):
    """Rescale image between 0 and 1"""

    img = img.astype(np.float32)

    min = np.min(img)
    max = np.max(img)

    if np.absolute(min-max) < 1.e-6:
        img.fill(0.5)
    else:
        img = (img - min)/(max - min)

    return img.astype(np.float32)

def hover(event):
    # if the mouse is over the scatter points
    if line.contains(event)[0]:
        # find out the index within the array from the event
        ind, = line.contains(event)[1]["ind"]
        # get the figure size
        w,h = fig.get_size_inches()*fig.dpi
        ws = (event.x > w/2.)*-1 + (event.x <= w/2.) 
        hs = (event.y > h/2.)*-1 + (event.y <= h/2.)
        # if event occurs in the top or right quadrant of the figure,
        # change the annotation box position relative to mouse.
        ab.xybox = (xybox[0]*ws, xybox[1]*hs)
        # make annotation box visible
        ab.set_visible(True)
        # place it at the position of the hovered scatter point
        ab.xy =(x[ind], y[ind])
        # set the image corresponding to that point
        if arr.shape[-1] == 1:
            im.set_data(arr[ind,:,:,0])
        else:
            im.set_data(arr[ind,:,:,:])
    else:
        #if the mouse is not over a scatter point
        ab.set_visible(False)
    fig.canvas.draw_idle()


if __name__ == "__main__":

    #SAVE_DATA = "//Desktop-sa1evjv/h/small_scans/96x96.npy"
    #SAVE_FILE = "//ads.warwick.ac.uk/shared/HCSS6/Shared305/Microscopy/Jeffrey-Ede/models/visualize_data/tsne_stem_downsampled_96x96.npy"

    #SAVE_DATA = "//Desktop-sa1evjv/h/small_scans-tem/96x96-tem.npy"
    #SAVE_FILE = "//ads.warwick.ac.uk/shared/HCSS6/Shared305/Microscopy/Jeffrey-Ede/models/visualize_data/tsne_tem_downsampled_96x96.npy"

    SAVE_DATA = "//Desktop-sa1evjv/h/wavefunctions_96x96/wavefunctions_single_n=3.npy"
    SAVE_FILE = "//ads.warwick.ac.uk/shared/HCSS6/Shared305/Microscopy/Jeffrey-Ede/models/visualize_data/tsne_wavefunctions_single_n=3.npy"

    USE_FRAC = .1 #Fraction of dataset to display. Decrease from 1 if slow.

    tsne = np.load(SAVE_FILE)

    x = tsne[:,0]
    y = tsne[:,1]

    #if "//Desktop-sa1evjv/h/small_scans-tem/96x96-tem.npy": #Remove a couple of outliers
    #    x0to1 = scale0to1(x)
    #    y0to1 = scale0to1(y)

    #    keep = (x0to1 > 0.3)*(x0to1 < 0.9)*(y0to1 > 0.2)*(y0to1 < 0.8)

    #    x = x[keep]
    #    y = y[keep]

    x = scale0to1(x)
    y = scale0to1(y)
    tsne = np.stack([x,y], axis=-1)

    arr = np.load(SAVE_DATA)

    if arr.shape[-1] == 1:
        arr = np.stack([scale0to1(x) for x in arr])
    else:
        std = np.std(arr)
        arr /= 7*std 
        arr += ( 0.5 - np.mean(arr) )
        arr = arr.clip(0., 1.)
        arr = np.concatenate( (arr[...,:1], np.zeros(list(arr.shape)[:-1] + [1]), arr[...,1:]), axis=-1 )

    fig = plt.figure()
    ax = fig.add_subplot(111)
    line, = ax.plot(x[:int(USE_FRAC*x.shape[0])],y[:int(USE_FRAC*y.shape[0])], ls="", marker="o", color="black", alpha=1, markersize=0.5)

    # create the annotations box
    if arr.shape[-1] == 1:
        im = OffsetImage(arr[0,:,:,0], zoom=1.0, cmap="gray", norm=mpl.colors.Normalize(vmin=0.,vmax=1.))
    else:
        im = OffsetImage(arr[0,:,:,:], zoom=1.0)
    xybox=(50., 50.)
    ab = AnnotationBbox(im, (0,0), xybox=xybox, xycoords='data',
            boxcoords="offset points",  pad=0.3,  arrowprops=dict(arrowstyle="-"))

    ax.add_artist(ab)

    ab.set_visible(False)


    # add callback for mouse moves
    fig.canvas.mpl_connect('motion_notify_event', hover)           
    plt.show()
