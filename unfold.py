import numpy as np
from skimage import morphology, util, segmentation
from skan import csr
from scipy.sparse import csgraph
from scipy import ndimage as ndi
from scipy import interpolate
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
    idxs_path_smoothed = ndi.gaussian_filter(idxs_path, sigma=(sigma, 0),
                                             mode='nearest')
    idxs_path_smoothed_int = np.round(idxs_path_smoothed).astype(int)
    # smoothing and rounding can result in repeated indices. We remove them.
    idxs_path_smoothed_unique = util.unique_rows(idxs_path_smoothed_int)
    return idxs_path_smoothed_unique


def source_id_volume(image, idxs, path):
    new_path_idxs = path_coordinates(idxs, path)
    new_path_ids = np.arange(1, new_path_idxs.shape[0] + 1)
    markers = np.zeros(image.shape, dtype=np.uint16)
    markers[tuple(new_path_idxs.T)] = new_path_ids
    sources = segmentation.watershed(image, markers=markers, compactness=1e10)
    # or: sources = ndi.distance_transform_edt(..., return_indices=True)
    # profile this.
    return sources, new_path_ids, new_path_idxs


def coord0_volume(sources, idxs):
    differences = np.diff(idxs, axis=0)
    distances_raw = np.sqrt(np.sum(differences ** 2, axis=1))
    distances = np.concatenate(([0, 0], distances_raw))
    return distances[sources]


def coord1_volume(thresholded_image):
    return ndi.distance_transform_edt(~thresholded_image)


def sample2d(coord0_volume, coord1_volume, image):
    xs = np.indices((np.max(coord0_volume).astype(int),
                     np.max(coord1_volume).astype(int)))
    xs_r = xs.reshape((2, -1)).T
    points = np.stack((coord0_volume.ravel(), coord1_volume.ravel()), axis=1)
    r = interpolate.griddata(points, image.ravel(), xs_r)
    return r.reshape(xs.shape[1:])
