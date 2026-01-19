import os
import os.path
from os.path import dirname, join
from typing import List

import cortex
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# import matplotlib.colormaps

# default matplotlib colors
cs_mpl = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

# a nice blue/red color
cblue = "#66ccff"
cred = "#cc0000"


def imshow_diverging(mat, clab="Mean response ($\\sigma$)", clab_size='medium', vabs_multiplier=1):
    vabs = np.nanmax(np.abs(mat)) * vabs_multiplier
    plt.imshow(mat, cmap=sns.diverging_palette(
        220, 29, as_cmap=True), vmin=-vabs, vmax=vabs)
    cb = plt.colorbar()
    # set tick label size
    # cb.ax.tick_params(labelsize=clab_size)
    cb.set_label(label=clab, size=clab_size)


def save_figs_to_single_pdf(filename):
    p = PdfPages(filename)
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]
    for fig in figs:
        fig.savefig(p, format="pdf", bbox_inches="tight")
    p.close()


def colorize(
    words: List[str],
    color_array,  # : np.ndarray[float],
    char_width_max=60,
    title: str = None,
    subtitle: str = None,
):
    """
    Colorize a list of words based on a color array.
    color_array
        an array of numbers between 0 and 1 of length equal to words
    """
    cmap = matplotlib.colormaps.get_cmap("viridis")
    # cmap = matplotlib.cm.get_cmap('viridis_r')
    template = (
        '<span class="barcode"; style="color: black; background-color: {}">{}</span>'
    )
    # template = '<span class="barcode"; style="color: {}; background-color: white">{}</span>'
    colored_string = ""
    char_width = 0
    for word, color in zip(words, color_array):
        char_width += len(word)
        color = matplotlib.colors.rgb2hex(cmap(color)[:3])
        colored_string += template.format(color, "&nbsp" + word + "&nbsp")
        if char_width >= char_width_max:
            colored_string += "</br>"
            char_width = 0

    if subtitle:
        colored_string = f"<h5>{subtitle}</h5>\n" + colored_string
    if title:
        colored_string = f"<h3>{title}</h3>\n" + colored_string
    return colored_string


def moving_average(a, n=3):
    assert n % 2 == 1, "n should be odd"
    diff = n // 2
    vals = []
    # calculate moving average in a window 2
    # (1, 4)
    for i in range(diff, len(a) + diff):
        l = i - diff
        r = i + diff + 1
        vals.append(np.mean(a[l:r]))
    return np.nan_to_num(vals)


def get_story_scores(val, expls, paragraphs):
    import imodelsx.util

    # mod = EmbDiffModule()
    scores_list = []
    for i in range(len(expls)):
        # for i in range(1):
        expl = expls[i].lower()
        text = paragraphs[i]
        words = text.split()
        if len(words) == 1:
            words.append(' ')
        if len(words) == 0:
            words = [' ', ' ']
        # print('words', words)

        ngrams = imodelsx.util.generate_ngrams_list(text.lower(), ngrams=3)
        ngrams = [words[0], words[0] + " " + words[1]] + ngrams

        # # embdiff-based viz
        # mod._init_task(expl)
        # neg_dists = mod(ngrams)
        # assert len(ngrams) == len(words) == len(neg_dists)
        # # neg_dists = scipy.special.softmax(neg_dists)
        # # plt.plot(neg_dists)
        # # plt.plot(moving_average(neg_dists, n=5))
        # neg_dists = moving_average(neg_dists, n=3)
        # neg_dists = (neg_dists - neg_dists.min()) / (neg_dists.max() - neg_dists.min())
        # neg_dists = neg_dists / 2 + 0.5 # shift to 0.5-1 range
        # s = neuro.sasc.viz.colorize(words, neg_dists, title=expl, subtitle=prompt)

        # validator-based viz
        probs = np.array(val.validate_w_scores(expl, ngrams))
        scores_list.append(probs)
    return scores_list


def heatmap(
    data,
    labels,
    xlab="Explanation for matching",
    ylab="Explanation for generation",
    clab="Fraction of matching ngrams",
    diverging=True,
    label_fontsize='medium',
):
    # plt.style.use('dark_background')
    plt.figure(figsize=(7, 6))
    if diverging:
        imshow_diverging(data, clab=clab, clab_size=label_fontsize)
    else:
        plt.imshow(data, cmap='Blues')
        cb = plt.colorbar()
        # cb.ax.tick_params(labelsize=clab_size)
        cb.set_label(label=clab, size=label_fontsize)
    plt.xticks(range(data.shape[0]), labels, rotation=90, fontsize="small")
    plt.yticks(range(data.shape[1]), labels, fontsize="small")
    plt.ylabel(ylab, fontsize=label_fontsize)
    plt.xlabel(xlab, fontsize=label_fontsize)
    plt.tight_layout()
    # plt.show()


def outline_diagonal(shape, color='gray', lw=1, block_size=1):
    for r in range(shape[0]):
        for c in range(shape[1]):
            # outline the diagonal with blocksize 1
            if block_size == 1 and r == c:
                plt.plot([r - 0.5, r + 0.5],
                         [c - 0.5, c - 0.5], color=color, lw=lw)
                plt.plot([r - 0.5, r + 0.5],
                         [c + 0.5, c + 0.5], color=color, lw=lw)
                plt.plot([r - 0.5, r - 0.5],
                         [c - 0.5, c + 0.5], color=color, lw=lw)
                plt.plot([r + 0.5, r + 0.5],
                         [c - 0.5, c + 0.5], color=color, lw=lw)
            if block_size == 2 and r == c and r % 2 == 0:
                rx = r + 0.5
                cx = c + 0.5
                plt.plot([rx - 1, rx + 1], [cx - 1, cx - 1],
                         color=color, lw=lw)
                plt.plot([rx - 1, rx + 1], [cx + 1, cx + 1],
                         color=color, lw=lw)
                plt.plot([rx - 1, rx - 1], [cx - 1, cx + 1],
                         color=color, lw=lw)
                plt.plot([rx + 1, rx + 1], [cx - 1, cx + 1],
                         color=color, lw=lw)
            if block_size == 3 and r == c and r % 3 == 0:
                rx = r + 1
                cx = c + 1
                plt.plot([rx - 1.5, rx + 1.5],
                         [cx - 1.5, cx - 1.5], color=color, lw=lw)
                plt.plot([rx - 1.5, rx + 1.5],
                         [cx + 1.5, cx + 1.5], color=color, lw=lw)
                plt.plot([rx - 1.5, rx - 1.5],
                         [cx - 1.5, cx + 1.5], color=color, lw=lw)
                plt.plot([rx + 1.5, rx + 1.5],
                         [cx - 1.5, cx + 1.5], color=color, lw=lw)


def plot_annotated_resp(
    voxel_num: int,
    word_chunks,
    voxel_resp,
    expl_voxel,
    start_times,
    end_times,
    stories_data_dict,
    expls,
    story_num,
    word_chunks_contain_example_ngrams,
    trim=5,
    annotate_texts=True,
    plot_key_ngrams=True,
):
    plt.figure(figsize=(11, 3))
    plt.plot(voxel_resp, color='black')

    # annotate top 5 voxel_resps with word_chunks
    if annotate_texts:
        texts = []
        top_5_resp_positions = np.argsort(voxel_resp)[::-1][:5]
        for i, resp_position in enumerate(top_5_resp_positions):
            plt.plot(resp_position,
                     voxel_resp[resp_position], "o", color="black")
            text = (
                " ".join(word_chunks[resp_position - 1])
                + "\n"
                + " ".join(word_chunks[resp_position])
            )
            texts.append(
                plt.annotate(
                    text, (resp_position,
                           voxel_resp[resp_position]), fontsize="x-small"
                )
            )

        # annotate bottom 5 voxel_resps with word_chunks
        bottom_5_resp_positions = np.argsort(voxel_resp)[:5]
        for i, resp_position in enumerate(bottom_5_resp_positions):
            plt.plot(resp_position,
                     voxel_resp[resp_position], "o", color="black")
            text = (
                " ".join(word_chunks[resp_position - 2])
                + "\n"
                + " ".join(word_chunks[resp_position - 1])
                + "\n"
                + " ".join(word_chunks[resp_position])
            )
            texts.append(
                plt.annotate(
                    text, (resp_position,
                           voxel_resp[resp_position]), fontsize="x-small"
                )
            )
        # adjustText.adjust_text(texts, arrowprops=dict(
            # arrowstyle="->", color="gray"))

    # plot key ngrams
    if plot_key_ngrams:
        i_start_voxel = start_times[voxel_num]
        i_end_voxel = end_times[voxel_num] + 1
        if i_end_voxel > len(voxel_resp):
            i_end_voxel = len(voxel_resp)
        idxs = np.arange(i_start_voxel, i_end_voxel)
        idxs_wc = np.where(word_chunks_contain_example_ngrams[idxs])[0]
        plt.plot(idxs, voxel_resp[idxs], color="C0", linewidth=2)
        plt.plot(idxs[idxs_wc], voxel_resp[idxs[idxs_wc]],
                 "^", color="C1", linewidth=2)

    # clean up plot and add in trim
    plt.grid(alpha=0.4, axis="y")
    xticks = np.array([start_times - trim, end_times - trim]).mean(axis=0)
    plt.xticks(xticks, expls, rotation=45, fontsize="x-small")

    for i, (start_time, end_time) in enumerate(zip(start_times - trim, end_times - trim)):
        if i == voxel_num:
            plt.axvspan(start_time, end_time, facecolor="C0", alpha=0.12)
        elif i % 2 == 0:
            plt.axvspan(start_time, end_time, facecolor="gray", alpha=0.08)
        else:
            plt.axvspan(start_time, end_time, facecolor="gray", alpha=0.0)
    plt.xlim((start_times[0] - trim, end_times[-1] - trim))

    plt.ylabel(
        f'"{expl_voxel}"\nvoxel\nresponse ($\\sigma_f$)', fontsize='x-small'
    )

    # plt.show()


def barplot_default(
        diag_means_list: List[np.ndarray],
        off_diag_means_list: List[np.ndarray],
        pilot_name, expls, annot_points=True, spread=50
):

    plt.figure(dpi=300)

    # raw inputs
    markers = ['o', 'x', '^']
    n = sum([len(diag_means) for diag_means in diag_means_list])
    x = np.arange(n) - n / 2
    offset = 0
    ms = 6
    for i in range(len(diag_means_list)):
        diag_means = diag_means_list[i]
        off_diag_means = off_diag_means_list[i]

        # plot individual points
        xp = x[offset:offset + len(diag_means)]
        offset += len(diag_means)
        plt.plot(1 + xp/spread, diag_means,
                 markers[i], color='C0', alpha=0.9, markersize=ms, markeredgewidth=2)
        plt.plot(2 + xp/spread, off_diag_means,
                 markers[i], color='C1', markersize=ms, markeredgewidth=2)

    # plot overarching bars
    # get mean of each row excluding the diagonal
    diag_mean = np.nanmean(np.concatenate(diag_means_list))
    off_diag_mean = np.nanmean(np.concatenate(off_diag_means_list))
    plt.bar(1, diag_mean, width=0.7, alpha=0.2, color='C0')
    plt.errorbar(1, diag_mean, yerr=np.nanstd(diag_means) / np.sqrt(len(diag_means)),
                 fmt='.', ms=0, color='black', elinewidth=3, capsize=5, lw=1)

    plt.bar(2, off_diag_mean, width=0.5, alpha=0.1, color='C1')
    plt.errorbar(2, off_diag_mean, yerr=np.nanstd(off_diag_means) / np.sqrt(len(off_diag_means)),
                 fmt='.', ms=0, color='black', elinewidth=3, capsize=5)

    plt.xticks([1, 2], ['Drive', 'Baseline'])
    plt.ylabel('Mean voxel response ($\\sigma_f$)')
    plt.grid(axis='y')

    # annotate the point with the highest mean
    if annot_points:
        kwargs = dict(
            arrowprops=dict(arrowstyle='->', color='#333'), fontsize='x-small', color='#333'
        )
        idx = np.argmax(diag_means)
        print(expls[idx])
        plt.annotate(f"{expls[idx]}", (1 + x[idx]/50, diag_means[idx]),
                     xytext=(1.1, diag_means[idx] + 0.1), **kwargs)

        # annotate the point with the second highest mean
        idx = np.argsort(diag_means)[-2]
        print(expls[idx])
        plt.annotate(f"{expls[idx]}", (1 + x[idx]/50, diag_means[idx]),
                     xytext=(1.1, diag_means[idx] + 0.1), **kwargs)

        # annotate the point with the lowest mean
        idx = np.argmin(diag_means)
        plt.annotate(f"{expls[idx]}", (1 + x[idx]/50, diag_means[idx]),
                     xytext=(1.1, diag_means[idx]), **kwargs)

    plt.tight_layout()
    print('mean', diag_mean - off_diag_mean)
    # plt.title(f'use_clusters={use_clusters}')
    # plt.title('Single voxel', y=0.9)
    plt.savefig(join(pilot_name[:pilot_name.index(
        '_')] + '_default_means.pdf'), bbox_inches='tight')


def barplot_interaction(
    diag_means_list, off_diag_means_list, diag_means_interaction_list, off_diag_means_interaction_list,
        pilot_name, spread=50
):
    plt.figure(dpi=300)

    n0 = sum([len(diag_means) for diag_means in diag_means_list])
    n1 = sum([len(off_diag_means) for off_diag_means in off_diag_means_list])
    n2 = sum([len(diag_means_interaction)
             for diag_means_interaction in diag_means_interaction_list])
    x0 = np.arange(n0) - n0 / 2
    x1 = np.arange(n1) - n1 / 2
    x2 = np.arange(n2) - n2 / 2
    offset0 = 0
    offset1 = 0
    offset2 = 0
    markers = ['o', 'x', '^']
    for i in range(len(diag_means_list)):

        diag_means = diag_means_list[i]
        off_diag_means = off_diag_means_list[i]
        diag_means_interaction = diag_means_interaction_list[i]
        off_diag_means_interaction = off_diag_means_interaction_list[i]

        # n = len(diag_means)
        m = markers[i]
        ms = 6
        me = 2
        plt.plot(0 + x0[offset0:offset0 + len(diag_means)] /
                 spread, diag_means, m, color='C0', markersize=ms, markeredgewidth=me)
        plt.plot(2 + x1[offset1:offset1 + len(off_diag_means)] /
                 spread, off_diag_means, m, color='C1', markersize=ms, markeredgewidth=me)
        plt.plot(1 + x2[offset2:offset2 + len(diag_means_interaction)]/spread,
                 diag_means_interaction, m, color='#5D3F6A', markersize=ms, markeredgewidth=me)

        offset0 += len(diag_means)
        offset1 += len(off_diag_means)
        offset2 += len(diag_means_interaction)

    diag_means = np.concatenate(diag_means_list)
    off_diag_means = np.concatenate(off_diag_means_list)
    diag_means_interaction = np.concatenate(diag_means_interaction_list)
    off_diag_means_interaction = np.concatenate(
        off_diag_means_interaction_list)
    diag_mean = np.nanmean(diag_means)
    off_diag_mean = np.nanmean(off_diag_means)
    diag_mean_interaction = np.nanmean(diag_means_interaction)
    off_diag_mean_interaction = np.nanmean(off_diag_means_interaction)
    plt.bar(0, diag_mean, width=0.6,
            label='Diagonal', alpha=0.2, color='C0')
    plt.errorbar(0, diag_mean, yerr=np.nanstd(diag_means) / np.sqrt(len(diag_means)),
                 fmt='.', label='Diagonal', ms=0, color='black', elinewidth=3, capsize=5, lw=1)
    plt.bar(2, off_diag_mean, width=0.6,
            label='Off-diagonal', alpha=0.2, color='C1')
    plt.errorbar(2, off_diag_mean, yerr=np.nanstd(off_diag_means) / np.sqrt(len(off_diag_means)),
                 fmt='.', label='Diagonal', ms=0, color='black', elinewidth=3, capsize=5)

    plt.bar(1, diag_mean_interaction, width=0.6,
            label='Diagonal', alpha=0.2, color='#5D3F6A')
    plt.errorbar(1, diag_mean_interaction, yerr=np.nanstd(diag_means_interaction) / np.sqrt(len(diag_means_interaction)),
                 fmt='.', label='Diagonal', ms=0, color='black', elinewidth=3, capsize=5, lw=1)

    plt.xticks([0, 1, 2], ['Drive single', 'Drive pair', 'Baseline'])
    plt.ylabel('Mean voxel response ($\\sigma_f$)')
    plt.grid(axis='y')

    plt.tight_layout()
    print('mean single', diag_mean - off_diag_mean)
    print('mean inter', diag_mean - off_diag_mean_interaction)
    print('median', np.nanmedian(diag_means_interaction) -
          np.nanmedian(off_diag_means_interaction))
    # move title down into plot
    # plt.title('Voxel interaction', y=0.9)
    # plt.title(f'use_clusters={use_clusters}')
    plt.savefig(join(pilot_name[:pilot_name.index(
        '_')] + '_interaction_means.pdf'), bbox_inches='tight')


def barplot_polysemantic(
    diag_means_list: List[np.ndarray], off_diag_means_list: List[np.ndarray],
    pilot_name, expls, annot_points=True, spread=60
):

    plt.figure(dpi=300)

    # raw inputs
    markers = ['X', '^']
    n = sum([len(diag_means) for diag_means in diag_means_list])
    x = np.arange(n) - n / 2
    offset = 0
    for i in range(len(diag_means_list)):
        diag_means = diag_means_list[i]
        off_diag_means = off_diag_means_list[i]

        # plot individual points
        print('offset', offset, len(diag_means))
        xp = x[offset:offset + len(diag_means)]
        offset += len(diag_means)

        m = markers[i]
        ms = 5
        for j in range(0, len(diag_means), 2):
            bigger = max(diag_means[j], diag_means[j + 1])
            smaller = min(diag_means[j], diag_means[j + 1])
            xj = xp[j] / spread
            plt.plot(1 + xj, bigger, m, color='C0', alpha=0.9, markersize=ms)
            plt.plot(1 + xj, smaller, m, color='C0', alpha=0.3, markersize=ms)
            plt.plot([1 + xj, 1 + xj],
                     [bigger, smaller], color='gray', alpha=0.3)
        plt.plot(2 + xp/spread, off_diag_means, m, color='gray', markersize=ms)

    # plot overarching bars
    # get mean of each row excluding the diagonal
    diag_mean = np.nanmean(np.concatenate(diag_means_list))
    off_diag_mean = np.nanmean(np.concatenate(off_diag_means_list))
    plt.bar(1, diag_mean, width=0.5, alpha=0.2, color='C0')
    plt.errorbar(1, diag_mean, yerr=np.nanstd(diag_means) / np.sqrt(len(diag_means)),
                 fmt='.', ms=0, color='black', elinewidth=3, capsize=5, lw=1)

    plt.bar(2, off_diag_mean, width=0.5, alpha=0.1, color='gray')
    plt.errorbar(2, off_diag_mean, yerr=np.nanstd(off_diag_means) / np.sqrt(len(off_diag_means)),
                 fmt='.', ms=0, color='black', elinewidth=3, capsize=5)

    plt.xticks([1, 2], ['Drive', 'Baseline'])
    plt.ylabel('Mean voxel response ($\\sigma$)')
    plt.grid(axis='y')

    # annotate the point with the highest mean
    if annot_points:
        kwargs = dict(
            arrowprops=dict(arrowstyle='->', color='#333'), fontsize='x-small', color='#333'
        )
        idx = np.argmax(diag_means)
        print(expls[idx])
        plt.annotate(f"{expls[idx]}", (1 + x[idx]/50, diag_means[idx]),
                     xytext=(1.1, diag_means[idx] + 0.1), **kwargs)

        # annotate the point with the second highest mean
        idx = np.argsort(diag_means)[-2]
        print(expls[idx])
        plt.annotate(f"{expls[idx]}", (1 + x[idx]/50, diag_means[idx]),
                     xytext=(1.1, diag_means[idx] + 0.1), **kwargs)

        # annotate the point with the lowest mean
        idx = np.argmin(diag_means)
        plt.annotate(f"{expls[idx]}", (1 + x[idx]/50, diag_means[idx]),
                     xytext=(1.1, diag_means[idx]), **kwargs)

    plt.tight_layout()
    print('mean', diag_mean - off_diag_mean)
    # plt.title(f'use_clusters={use_clusters}')
    # plt.title('Polysemantic', y=0.9)


def stories_barplot(story_scores_df):
    story_scores_df = story_scores_df.melt(id_vars='story', value_vars=[
        'driving', 'baseline'], var_name='condition', value_name='mean')
    story_scores_df = story_scores_df.sort_values(by='story')
    sns.barplot(data=story_scores_df, x='story', y='mean', hue='condition')
    plt.ylabel('Mean voxel response ($\\sigma_f$)')


def _save_flatmap(vals, subject, fname_save, clab=None, with_rois=True, cmap='RdBu_r', with_borders=False, show=False, vabs=None):
    if vabs is None:
        vabs = max(np.abs(vals))

    # cmap = sns.diverging_palette(12, 210, as_cmap=True)
    # cmap = sns.diverging_palette(16, 240, as_cmap=True)

    vol = cortex.Volume(
        vals, 'UT' + subject, xfmname=f'UT{subject}_auto', vmin=-vabs, vmax=vabs, cmap=cmap)

    cortex.quickshow(vol,
                     with_rois=with_rois,
                     with_labels=False,
                     with_borders=with_borders,
                     with_colorbar=clab == None,  # if not None, save separate cbar
                     )
    os.makedirs(dirname(fname_save), exist_ok=True)
    plt.savefig(fname_save)
    if not show:
        plt.close()

    # save cbar
    norm = Normalize(vmin=-vabs, vmax=vabs)
    # need to invert this to match above
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig, ax = plt.subplots(figsize=(5, 0.35))
    if clab:
        cbar = plt.colorbar(sm, cax=ax, orientation='horizontal')
        cbar.set_label(clab, fontsize='x-large')
        plt.savefig(fname_save.replace('flatmap', '').replace(
            '.', 'cbar.').replace('.png', '.pdf'), bbox_inches='tight')
    if not show:
        plt.close()
