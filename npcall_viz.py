import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as pt
from matplotlib.patches import Ellipse, FancyArrowPatch, ArrowStyle
from pathlib import Path

def calculate_decision_function(inputs, means, covariances):
    for i, (mean, covariance) in enumerate(zip(means, covariances)):
        L = np.linalg.cholesky(covariance)
        z = np.linalg.solve(L, (inputs - mean).transpose())
        s = (z ** 2).sum(axis = 0) + 2 * np.log(L.diagonal()).sum()
        if i == 0: scores = s
        else: scores -= s
    return scores, np.where(
        scores > 0,
        1 / (1 + np.exp(-scores)),
        np.exp(scores) / (1 + np.exp(scores))
    )

def visualize_npcall_distribution(
    inputs,
    outputs,
    targets,
    means,
    covariances,
    title = None,
    export_path = Path('.'),
    margin = [ 0.025, 0.975 ],
    confidence = [ 0.95 ]
):
    fig = pt.figure(figsize = (10, 10), dpi = 100)
    ax = fig.add_subplot(1,1,1)
    
    # evaluate statistics
    scores, probs = calculate_decision_function(inputs, means, covariances)
    failures = outputs != targets
    nocalls = (probs > margin[0]) & (probs < margin[1])
    labels = np.unique(targets)
    
    # draw scatter plot of data
    colors = ['green','red']
    for color, selection in zip(colors, [targets == label for label in labels]):
        ax.scatter(inputs[selection,0], inputs[selection,1], marker = ',', s = 3, c = color, alpha = 0.2)
    
    # draw failed calls and no calls
    errors = failures & ~nocalls
    ax.scatter(inputs[nocalls,0], inputs[nocalls,1], marker = 's', s = 5, c = 'gray')
    ax.scatter(inputs[errors ,0], inputs[errors ,1], marker = 'x', s = 12, c = 'blue', alpha = 0.5)
    xmin, ymin = np.floor(inputs.min(axis = 0))
    xmax, ymax = np.ceil (inputs.max(axis = 0))
    #xmax = ymax = max(xmax, ymax)
    #xmin = ymin = min(xmin, ymin)
    
    # set labels
    ax.set_xlabel('channel 0')
    ax.set_ylabel('channel 1')
    ax.set_xlim(xmin, xmax + 0.5)
    ax.set_ylim(ymin, ymax + 0.5)
    ax.set_xticks(np.arange(xmin, xmax + 1))
    ax.set_yticks(np.arange(ymin, ymax + 1))
    ax.legend(['Type 0', 'Type 1', 'No call', 'Error'])
    if title is not None:
        ax.set_title(title)
    
    # draw contour
    size = 201, 201
    xx, yy = np.meshgrid(
        np.linspace(xmin, xmax + 0.5, size[1]),
        np.linspace(ymin, ymax + 0.5, size[0]),
    )
    tt = np.stack((xx.ravel(), yy.ravel()), axis = 1)
    _, zz = calculate_decision_function(tt, means, covariances)
    cs = ax.contour(xx, yy, zz.reshape(size), [margin[0], 0.5, margin[1]], alpha = 0.75, colors = 'k')
    ax.clabel(cs, fontsize = 10)
    
    # draw Gaussian models
    stdevs  = []
    vectors = []
    for mean, covariance, color in zip(means, covariances, colors):
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
                edgecolor = color
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
    ax.text(
        (xmax + 0.5) * (1 - 0.985) + 0.985 * xmin,
        (ymax + 0.5) * (1 - 0.015) + 0.015 * ymin,
        '\n'.join([
            'Call rate = {:.1f}%',
            'Accuracy = {:.1f}%',
            'Accuracy (inc. no calls) = {:.1f}%',
            'Accuracy (exc. no calls) = {:.1f}%',
            'Angle b/w clusters = {:.1f} degrees',
            'Separation index = {:.2f}',
            'Mean.0 = {:.0f}, {:.0f}',
            'Mean.1 = {:.0f}, {:.0f}',
            'Stdev.0 = {:.2f}',
            'Stdev.1 = {:.2f}',
        ]).format(
            (1 - sum(nocalls) / len(inputs)) * 100,
            *accuracies,
            np.degrees(np.arccos(np.clip(np.dot(*vectors), -1, 1))),
            w.dot(means[0] - means[1]) ** 2 / w.dot(covariances.sum(axis = 0)).dot(w),
            2 ** means[0][0], 2 ** means[0][1],
            2 ** means[1][0], 2 ** means[1][1],
            *stdevs
        ),
        linespacing = 1.5,
        ha = 'left',
        va = 'top',
    )
    ax.set_aspect('equal')
    fig.savefig(str(export_path), dpi = 200)
    # fig.show()
    pt.close(fig)
