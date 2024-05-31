"""
Modules for visualization.
Copied and modified from the following codes:
https://github.com/drscotthawley/aeiou/blob/main/aeiou/viz.py
"""

import torch
from PIL import Image
import numpy as np
import torchaudio.transforms as T
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from einops import rearrange


def mel_spectrogram(waveform, power=2.0, sample_rate=48000, db=False, n_fft=1024, n_mels=128):
    "calculates data array for mel spectrogram (in however many channels)"
    win_length = None
    hop_length = n_fft // 2  # 512

    mel_spectrogram_op = T.MelSpectrogram(
        sample_rate=sample_rate, n_fft=n_fft, win_length=win_length,
        hop_length=hop_length, center=True, pad_mode="reflect", power=power,
        norm='slaney', onesided=True, n_mels=n_mels, mel_scale="htk")

    melspec = mel_spectrogram_op(waveform.float())
    if db:
        amp_to_db_op = T.AmplitudeToDB()
        melspec = amp_to_db_op(melspec)

    return melspec


def spectrogram_image(
    spec,
    db_range=[35, 120],
    figsize=(6, 3),  # size of plot (if justimage==False)
):
    from librosa import power_to_db

    fig = plt.figure(figsize=figsize, dpi=100)
    canvas = FigureCanvasAgg(fig)
    axs = fig.add_subplot()
    spec = spec.squeeze()
    im = axs.imshow(power_to_db(spec), origin='lower', aspect='auto', vmin=db_range[0], vmax=db_range[1])

    axs.axis('off')
    plt.tight_layout()

    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    im = Image.fromarray(rgba)

    b = 15  # border size
    im = im.crop((b, b, im.size[0] - b, im.size[1] - b))

    return im


def audio_spectrogram_image(waveform, power=2.0, sample_rate=48000, db=False, db_range=[35, 120], figsize=(6, 3)):
    """
    Wrapper for calling above two routines at once, does Mel scale; Modified from PyTorch tutorial
    https://pytorch.org/tutorials/beginner/audio_feature_extractions_tutorial.html
    """
    melspec = mel_spectrogram(waveform, power=power, db=db, sample_rate=sample_rate)
    melspec = melspec[0]  # TODO: only left channel for now
    return spectrogram_image(melspec, db_range=db_range, figsize=figsize)


def tokens_spectrogram_image(
    tokens,                # the embeddings themselves (in some diffusion codes these are called 'tokens')
    aspect='auto',         # aspect ratio of plot
    title='Embeddings',    # title to put on top
    ylabel='index',        # label for y axis of plot
    cmap='coolwarm',       # colormap to use. (default used to be 'viridis')
    symmetric=True,        # make color scale symmetric about zero, i.e. +/- same extremes
    figsize=(8, 4),       # matplotlib size of the figure
    dpi=100,               # dpi of figure
):
    "for visualizing embeddings in a spectrogram-like way"
    embeddings = rearrange(tokens, 'b d n -> (b n) d')  # expand batches in time
    vmin, vmax = None, None
    if symmetric:
        vmax = torch.abs(embeddings).max()
        vmin = -vmax

    fig = plt.figure(figsize=figsize, dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    if symmetric:
        subtitle = f'min={embeddings.min():0.4g}, max={embeddings.max():0.4g}'
        ax.set_title(title + '\n')
        ax.text(x=0.435, y=0.9, s=subtitle, fontsize=11, ha="center", transform=fig.transFigure)
    else:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('time frame (samples, in batches)')

    im = ax.imshow(embeddings.cpu().numpy().T, origin='lower', aspect=aspect, interpolation='none',
                   cmap=cmap, vmin=vmin, vmax=vmax)  # .T because numpy is x/y 'backwards'
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    return Image.fromarray(rgba)


def project_down(
    tokens,     # batched high-dimensional data with dims (b,d,n)
    proj_dims=3,     # dimensions to project to
):
    """this projects to lower dimenions, grabbing the first _`proj_dims`_ dimensions"""
    A = rearrange(tokens, 'b d n -> (b n) d')  # put all the vectors into the same d-dim space
    if A.shape[-1] > proj_dims:
        # PCA
        (U, S, V) = torch.pca_lowrank(A)
        proj_data = torch.matmul(A, V[:, :proj_dims])  # this is the actual PCA projection step
    else:
        proj_data = A

    return torch.reshape(proj_data, (tokens.size()[0], -1, proj_dims))  # put it in shape [batch, n, proj_dims]


def point_cloud(
    tokens,                  # embeddings / latent vectors. shape = (b, d, n)
    color_scheme='batch',    # 'batch': group by sample; integer n: n groups, sequentially,  otherwise color sequentially by time step
    output_type='wandbobj',  # plotly | points | wandbobj.  NOTE: WandB can do 'plotly' directly!
    mode='markers',    # plotly scatter mode.  'lines+markers' or 'markers'
    size=3,            # size of the dots
    # if mode='lines+markers', plotly line specifier. cf.
    # https://plotly.github.io/plotly.py-docs/generated/plotly.graph_objects.scatter3d.html#plotly.graph_objects.scatter3d.Line
    line=dict(color='rgba(10,10,10,0.01)'),
    ds_preproj=1,         # EXPERIMENTAL: downsampling factor before projecting  (1=no downsampling). Could screw up colors
    ds_preplot=1,         # EXPERIMENTAL: downsampling factor before plotting (1=no downsampling). Could screw up colors
    colormap=None,        # valid color map to use, None=defaults
    darkmode=False,       # dark background, white fonts
    layout_dict=None,      # extra plotly layout options such as camera orientation
    **kwargs,             # anything else to pass along
):
    """returns a 3D point cloud of the tokens"""

    import plotly.graph_objects as go
    import wandb

    if ds_preproj != 1:
        tokens = tokens[torch.randperm(tokens.size()[0])]  # EXPERIMENTAL: to 'correct' for possible weird effects of downsampling
        tokens = tokens[::ds_preproj]

    data = project_down(tokens, **kwargs).cpu().numpy()
    if data.shape[-1] < 3:  # for data less than 3D, embed it in 3D
        data = np.pad(data, ((0, 0), (0, 0), (0, 3 - data.shape[-1])), mode='constant', constant_values=0)

    bytime = False
    points = []
    if color_scheme == 'batch':  # all dots in same batch index same color, each batch-index unique (almost)
        ncolors = data.shape[0]
        cmap, norm = cm.tab20, Normalize(vmin=0, vmax=ncolors)
    elif isinstance(color_scheme, int) or color_scheme.isnumeric():  # n groups, by batch-indices, sequentially
        ncolors = int(color_scheme)
        cmap, norm = cm.tab20, Normalize(vmin=0, vmax=ncolors)
    else:                                                    # time steps match up
        bytime, ncolors = True, data.shape[1]
        cmap, norm = cm.viridis, Normalize(vmin=0, vmax=ncolors)

    cmap = cmap if colormap is None else colormap  # overwrite default cmap with user choice if given

    points = []
    for bi in range(data.shape[0]):  # batch index
        if color_scheme == 'batch':
            [r, g, b, _] = [int(255 * x) for x in cmap(norm(bi + 1))]
        elif isinstance(color_scheme, int) or color_scheme.isnumeric():
            grouplen = data.shape[0] // (ncolors)
            # if debug: print(f"bi, grouplen, bi//grouplen = ",bi, grouplen, bi//grouplen)
            [r, g, b, _] = [int(255 * x) for x in cmap(norm(bi // grouplen))]
            # if debug: print("r,g,b = ",r,g,b)
        for n in range(data.shape[1]):    # across time
            if bytime:
                [r, g, b, _] = [int(255 * x) for x in cmap(norm(n))]
            points.append([data[bi, n, 0], data[bi, n, 1], data[bi, n, 2], r, g, b])  # include dot colors with point coordinates

    point_cloud = np.array(points)

    if output_type == 'points':
        return point_cloud
    elif output_type == 'plotly':
        fig = go.Figure(data=[go.Scatter3d(
            x=point_cloud[::ds_preplot, 0], y=point_cloud[::ds_preplot, 1], z=point_cloud[::ds_preplot, 2],
            marker=dict(size=size, color=point_cloud[:, 3:6]),
            mode=mode,
            # show batch index and time index in tooltips:
            text=[f'bi: {i * ds_preplot}, ti: {j}' for i in range(data.shape[0] // ds_preplot) for j in range(data.shape[1])],
            line=line,
        )])
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))  # tight layout
        if darkmode:
            fig.layout.template = 'plotly_dark'
            if isinstance(darkmode, str):   # 'rgb(12,15,24)'gradio margins in dark mode
                fig.update_layout(paper_bgcolor=darkmode)
        if layout_dict:
            fig.update_layout(**layout_dict)

        return fig
    else:
        return wandb.Object3D(point_cloud)


def pca_point_cloud(
    tokens,                  # embeddings / latent vectors. shape = (b, d, n)
    color_scheme='batch',    # 'batch': group by sample, otherwise color sequentially
    output_type='wandbobj',  # plotly | points | wandbobj.  NOTE: WandB can do 'plotly' directly!
    mode='markers',    # plotly scatter mode.  'lines+markers' or 'markers'
    size=3,            # size of the dots
    # if mode='lines+markers', plotly line specifier. cf.
    # https://plotly.github.io/plotly.py-docs/generated/plotly.graph_objects.scatter3d.html#plotly.graph_objects.scatter3d.Line
    line=dict(color='rgba(10,10,10,0.01)'),
    **kwargs,
):
    return point_cloud(tokens, color_scheme=color_scheme, output_type=output_type,
                       mode=mode, size=size, line=line, **kwargs)
