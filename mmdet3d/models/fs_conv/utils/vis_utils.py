import open3d as o3d
class vis_pointcloud:
    def __init__(self,use_vis):
        self.use_vis=use_vis
        if self.use_vis==0:
            return
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="scene",width=640,height=480,left=50)
        render_option=self.vis.get_render_option()
        render_option.point_size=0.5

    def update(self,points,points_color):
        if self.use_vis==0:
            return
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors =  o3d.utility.Vector3dVector(points_color/255)
        self.vis.add_geometry(pcd)
        self.vis.poll_events()
        self.vis.update_renderer()
    
    def run(self):
        if self.use_vis==0:
            return
        self.vis.run()


class Vis_color:
    def __init__(self,use_vis):
        self.use_vis=use_vis
        if use_vis==0:
            return
        self.vis_image = o3d.visualization.Visualizer()
        self.vis_image.create_window(window_name="scene",width=320,height=240,left=720)

    def update(self,color_image):
        if self.use_vis==0:
            return
        geometry_image=o3d.geometry.Image(color_image)
        self.vis_image.add_geometry(geometry_image)
        self.vis_image.poll_events()
        self.vis_image.update_renderer()
        geometry_image.clear()
        