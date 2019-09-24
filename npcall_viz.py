import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as pt
from matplotlib.patches import Ellipse, FancyArrowPatch, ArrowStyle
from pathlib import Path

def calculate_standard_dqc(at, cg):
    good_at = np.sum(at['contrast'] > cg['contrast'].mean() + 2 * cg['contrast'].std())
    good_cg = np.sum(cg['contrast'] < at['contrast'].mean() - 2 * at['contrast'].std())
    return good_at / len(at), good_cg / len(cg)

# print('SD(AT,CG) = ({:.4f}, {:.4f})'.format(*calc_standard_dqc(cntrl_at, cntrl_cg)))
# print('SD(AT,CG) = ({:.4f}, {:.4f})'.format(*calc_standard_dqc(treat_at, treat_cg)))

def calculate_decision_function(inputs, means, covariances):
    for i, (mean, covariance) in enumerate(zip(means, covariances)):
        L = np.linalg.cholesky(covariance)
        z = np.linalg.solve(L, (inputs - mean).transpose())
        s = (z ** 2).sum(axis = 0) + 2 * np.log(L.diagonal()).sum()
        if i == 0: scores = s
        else: scores -= s
    
    probs = []
    for score in scores:
        if score > 0:
            t = np.exp(-score)
            probs.append(1 / (1 + t))
        else:
            t = np.exp(score)
            probs.append(t / (1 + t))

    return scores, np.array(probs)

#TODO
# configurable xylimits: { auto, manual } x { x, y } x { min, max }

def visualize_npcall_distribution(
    inputs,
    outputs,
    targets,
    labels,
    means,
    covariances,
    title = None,
    export_path = Path('.'),
    margin = [ 0.025, 0.975 ],
    confidence = [ 0.95 ],
    figsize = (8,8),
    fontsize = 12,
):
    fig = pt.figure(figsize = figsize)
    ax = fig.add_subplot(1,1,1)
    
    # evaluate statistics
    scores, probs = calculate_decision_function(inputs, means, covariances)
    failures = outputs != targets
    nocalls  = (probs > margin[0]) & (probs < margin[1])
    
    # draw scatter plot of data
    for label, color in labels:
        selection = targets == label
        ax.scatter(
            inputs[selection,0],
            inputs[selection,1],
            s = 3,
            c = color,
            marker = ',',
            alpha = 0.2
        )
    
    # draw failed calls and no calls
    errors = failures & ~nocalls
    ax.scatter(inputs[nocalls,0], inputs[nocalls,1], marker = 's', s = 5 , c = 'gray')
    ax.scatter(inputs[errors ,0], inputs[errors ,1], marker = 'x', s = 12, c = 'blue', alpha = 0.5)
    xmin, ymin = np.floor(inputs.min(axis = 0))
    xmax, ymax = np.ceil (inputs.max(axis = 0))
#    xmax = ymax = max(xmax, ymax)
#    xmin = ymin = min(xmin, ymin)
    
    # set labels
    ax.set_xlabel('channel {}'.format(labels[0][0]), fontsize = fontsize)
    ax.set_ylabel('channel {}'.format(labels[1][0]), fontsize = fontsize)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(np.arange(xmin, xmax + 1))
    ax.set_yticks(np.arange(ymin, ymax + 1))
#    ax.legend([
#        '{}-type'.format(labels[0][0]),
#        '{}-type'.format(labels[1][0]),
#        'No call',
#        'Error'
#    ])
    if title is not None:
        ax.set_title(title, fontsize = fontsize)
    
    # draw contour
    size = 501, 501
    xx, yy = np.meshgrid(
        np.linspace(xmin, xmax, size[1]),
        np.linspace(ymin, ymax, size[0]),
    )
    tt = np.stack((xx.ravel(), yy.ravel()), axis = 1)
    _, zz = calculate_decision_function(tt, means, covariances)
    cs = ax.contour(xx, yy, zz.reshape(size), [margin[0], 0.5, margin[1]], alpha = 0.75, colors = 'k')
    ax.clabel(cs, fontsize = 10)
    
    # draw Gaussian models
    stdevs  = []
    vectors = []
    indexes = np.argsort([ np.arctan2(*mean[::-1]) for mean in means ])
    for mean, covariance, idx in zip(means, covariances, indexes):
        eigvals, eigvecs = np.linalg.eig(covariance)
        indexes = eigvals.argsort()[::-1]
        eigvals = eigvals[indexes]
        eigvecs = eigvecs[:, indexes]
        if all(np.sign(eigvecs[:,0]) == [-1, -1]):
            eigvecs[:,0] *= -1
        vectors.append(eigvecs[:,0])
        stdevs .append(np.sqrt(eigvals[0]))
        for factor in [ ss.chi2.isf(alpha, 2) for alpha in 1 - np.array(confidence) ]:
            ax.add_patch(Ellipse(
                xy     = mean,
                width  = np.sqrt(eigvals[0] * factor) * 2,
                height = np.sqrt(eigvals[1] * factor) * 2,
                angle  = np.degrees(np.arctan2(*eigvecs[::-1,0])),
                fill   = False,
                edgecolor = labels[idx][1]
            ))
        for diff in (np.sqrt(eigvals) * eigvecs).transpose():
            ax.add_patch(FancyArrowPatch(
                posA = mean,
                posB = mean + diff,
                arrowstyle = ArrowStyle.Simple(
                    head_width  = 8,
                    tail_width  = 3,
                    head_length = 8,
                ),
                facecolor = 'black',
                edgecolor = 'white',
            ))    
        ax.plot(*mean, '*', color = 'yellow', markersize = 20, markeredgecolor = 'black')
    
    # annotate statistics
    w = np.linalg.solve(covariances.sum(axis = 0), means[0] - means[1])
    accuracies = 100 - np.array([
        sum(failures) / len(inputs),
        sum(failures |  nocalls) / len(inputs),
        sum(failures & ~nocalls) / (len(inputs) - sum(nocalls)), 
    ]) * 100
    
    # calcuate standard DQC
    si = np.exp(inputs)
    contrast = (si[:,0] - si[:,1]) / (si[:,0] + si[:,1])
    cg = contrast[targets == labels[0][0]]
    at = contrast[targets == labels[1][0]]
    good_at = np.count_nonzero(at < cg.mean() - 2 * cg.std()) / len(at)
    good_cg = np.count_nonzero(cg > at.mean() + 2 * at.std()) / len(cg)
    
    ax.text(
        xmax * (1 - 0.985) + 0.985 * xmin,
        ymax * (1 - 0.015) + 0.015 * ymin,
        '\n'.join([
            'Call rate = {:.1f}%',
            'Accuracy = {:.1f}%',
            'Accuracy (inc. no calls) = {:.1f}%',
            'Accuracy (exc. no calls) = {:.1f}%',
            'DQC(CG,AT) = {:.2f}, {:.2f}',
            'Angle b/w clusters = {:.1f} degrees',
            'Separation index = {:.2f}',
            'Mean.0 = {:.0f}, {:.0f}',
            'Mean.1 = {:.0f}, {:.0f}',
            'Stdev.0 = {:.2f}',
            'Stdev.1 = {:.2f}',
        ]).format(
            (1 - sum(nocalls) / len(inputs)) * 100,
            *accuracies,
            good_cg, good_at,
            np.degrees(np.arccos(np.clip(np.dot(*vectors), -1, 1))),
            w.dot(means[0] - means[1]) ** 2 / w.dot(covariances.sum(axis = 0)).dot(w),
            2 ** means[0][0], 2 ** means[0][1],
            2 ** means[1][0], 2 ** means[1][1],
            *stdevs
        ),
        linespacing = 1.5,
        ha = 'left',
        va = 'top',
        fontsize = fontsize,
    )
    ax.set_aspect('equal')
    fig.set_tight_layout(True)
    fig.savefig(str(export_path), dpi = 200)
    # fig.show()
    pt.close(fig)
    
    fig = pt.figure(figsize = figsize)
    ax = fig.add_subplot(1,1,1)
    ax.hist(cg, bins = 201, range = (-1, 1), color = labels[0][1], alpha = 0.5, density = True)
    ax.hist(at, bins = 201, range = (-1, 1), color = labels[1][1], alpha = 0.5, density = True)
    ax.set_xlabel('Contrast', fontsize = fontsize)
    ax.set_ylabel('Probability density', fontsize = fontsize)
    ax.set_title('DQC (CG = {:.2f} / AT = {:.2f})'.format(good_cg, good_at), fontsize = fontsize)
    ax.legend(['{}-type'.format(labels[0][0]), '{}-type'.format(labels[1][0])])
    fig.set_tight_layout(True)
    fig.savefig(str(export_path.parent / 'contrast_distr.png'), dpi = 200)
    pt.close(fig)
