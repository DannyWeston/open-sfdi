import open3d as o3d
import numpy as np

class PointCloudVisualiser:
    def __init__(self):
        pass

        self.__pointclouds = dict()
        self.__colours = dict()

        self.__visualiser = o3d.visualization.Visualizer()
        
    def add_numpy(self, name, np_cloud, colours=None):
        self.add_pointcloud(name, from_numpy(np_cloud), colours=colours)

    def add_pointcloud(self, name, cloud, colours=None):
        self.__pointclouds[name] = cloud

        if colours is not None:
            self.__pointclouds[name].colors = o3d.utility.Vector3dVector(colours)

    def filter_pointcloud(self, name, neighbours=75, std_ratio=0.75):
        cl, inliners = self.__pointclouds[name].remove_statistical_outlier(nb_neighbors=neighbours, std_ratio=std_ratio)
        self.__pointclouds[name] = self.__pointclouds[name].select_by_index(inliners)

        if name in self.__colours:
            self.__pointclouds.colors = self.__pointclouds.colors.select_by_index(inliners)

    def remove_pointcloud(self, name):
        self.__pointclouds.pop(name, None)
        self.__colours.pop(name, None)

    def get_pointcloud(self, name):
        if name in self.__pointclouds:
            return self.__pointclouds[name]
        
        return None

    def trans_pointcloud(self, name, x, y, z):
        if name in self.__pointclouds:
            self.__pointclouds[name].translate([x, y, z])

    def mask_pointcloud(self, name, mask):
        indices = np.where(mask.flatten())[0]
        self.__pointclouds[name] = self.__pointclouds[name].select_by_index(indices)

        if name in self.__colours:
            self.__colours[name] = self.__colours[name][indices]

    def align_pointclouds(self, target, names):
        for name in names:
            _, self.__pointclouds[name] = align_pcs(self.__pointclouds[target], self.__pointclouds[name])

    def show(self):
        self.__visualiser.create_window()

        for name in self.__pointclouds:
            self.__visualiser.add_geometry(self.__pointclouds[name])
        
        # opt = self.__visualiser.get_render_option()
        # opt.show_coordinate_frame = True

        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01, origin=[0, 0, 0])
        self.__visualiser.add_geometry(axis)

        self.__visualiser.run()

        self.__visualiser.clear_geometries()
        self.__visualiser.poll_events()
        self.__visualiser.update_renderer()
        self.__visualiser.destroy_window()

def rotate_pointcloud(pc, x, y, z, center=(0.0, 0.0, 0.0)):
    R = pc.get_rotation_matrix_from_xyz((x, y, z))
    pc.rotate(R, center=center)

    return pc

def save_cloud(pc, path):
    o3d.io.write_point_cloud(path, pc)

def from_numpy(arr):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(arr)

    return cloud

def chamfer_distance(pc1: o3d.geometry.PointCloud, pc2: o3d.geometry.PointCloud):
    return None

def load_mesh_as_pc(path, samples=10000):
    mesh = o3d.io.read_triangle_mesh(path)
    return mesh.sample_points_poisson_disk(samples)

def compute_fpfh(pcd, voxel_size):
    radius_feature = voxel_size * 5
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

def downsample_pc(pc, voxel_size):
    return pc.voxel_down_sample(voxel_size)

def global_registration(source, target, voxel_size):
    distance_threshold = voxel_size * 0.5
    
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source, target, 
        compute_fpfh(source, voxel_size), compute_fpfh(target, voxel_size),
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    
    return result

def icp(source, target, voxel_size):
    distance_threshold = voxel_size * 0.4

    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))
    
    return result

def align_pcs(pc, pc2, voxel_size=1.0):
    # Preprocess all point clouds
    pc = downsample_pc(pc, voxel_size)
    pc2 = downsample_pc(pc2, voxel_size)
    
    # Align pcd2 to pcd1 (Blender model as reference)
    transform = global_registration(pc2, pc, voxel_size)

    pc2.transform(transform.transformation)
    
    icp_transform = icp(pc2, pc, voxel_size)
    pc.transform(icp_transform.transformation)

    return pc, pc2