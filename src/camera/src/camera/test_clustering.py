

part_transform = state_to_matrix( numpy.loadtxt(part_tf_path,delimiter=",") )
            
stl_path = path + rosparam.get_param("/stl_params/directory_path") + \
                    "/" + rosparam.get_param("/stl_params/name") + ".stl"
logger.info("Reading stl. Path: {0}".format(stl_path))
self.mesh = open3d.io.read_triangle_mesh(stl_path)
logger.info("Stl read. Generating PointCloud from stl")
self.mesh = self.mesh.transform(part_transform)

# Point cloud of STL surface only
filters = rospy.get_param("/stl_params").get("filters")
self.stl_cloud = self.mesh.sample_points_poisson_disk(number_of_points=10000)
dot_products = numpy.asarray(self.stl_cloud.normals)[:,2]
dot_products[numpy.where(dot_products>1)[0]] = 1
dot_products[numpy.where(dot_products<-1)[0]] = -1
surface_indices = numpy.where( numpy.arccos(dot_products) < filters.get("max_angle_with_normal") )[0]
self.stl_cloud = self.stl_cloud.select_by_index(surface_indices)