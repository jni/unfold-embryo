import numpy as np
from skimage import morphology
from skan import csr
from scipy.sparse import csgraph
import networkx as nx



def define_mesoderm_axis(thresholded_image, *, spacing=1):
    skeleton = morphology.skeletonize_3d(thresholded_image)
    graph, idxs, degrees = csr.skeleton_to_csgraph(skeleton, spacing=spacing)
    distances, paths = csgraph.shortest_path(graph, return_predecessors=True)
    maxdist = np.nonzero(distances ==
                         np.max(distances[np.isfinite(distances)]))
    source, target = np.transpose(maxdist)[0]
    maxpath = nx.shortest_path(nx.from_scipy_sparse_matrix(graph),
                               source, target, weight='weight')
    return graph, idxs, maxpath


def path_coordinates(idxs, path, sigma=50):
    idxs_path = idxs[path]
