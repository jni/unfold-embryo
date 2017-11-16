import numpy as np
from skimage import morphology
from skan import csr
from scipy.sparse import csgraph

def define_mesoderm_axis(thresholded_image, *, spacing=1):
    skeleton = morphology.skeletonize_3d(thresholded_image)
    graph, idxs, degrees = csr.skeleton_to_csgraph(skeleton, spacing=spacing)
    distances, paths = csgraph.shortest_path(graph, return_predecessors=True)
    maxdist = np.nonzero(distances ==
                         np.max(distances[np.isfinite(distances)]))[0]
    return maxdist