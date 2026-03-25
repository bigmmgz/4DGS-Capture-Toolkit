bl_info = {
    "name": "4DGS Capture Toolkit",
    "blender": (4, 0, 0),
    "category": "Object",
    "version": (1, 0, 0),
    "author": "X (2025)",
    "description": "Multi-view capture and COLMAP dataset generation for 3D/4D Gaussian Splatting.",
    "location": "View3D > N-panel > 4DGS Tool",
    "support": 'COMMUNITY',
}

import bpy
import bmesh
import collections
import math
import mathutils
import os
import random
import numpy as np


# Paths and constants

ADDON_DIR = os.path.dirname(os.path.abspath(__file__))
ASSET_BLEND_PATH = os.path.join(ADDON_DIR, "assets", "predefined_objects.blend")
CAMERA_ARRAY_COLLECTION = "4DGS Data Tool"
VALID_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.exr'}
FORMAT_TO_EXTENSION = {
    "bmp": "bmp",
    "file_output": "exr",
    "jpeg": "jpg",
    "jp2": "jp2",
    "openexr": "exr",
    "png": "png",
    "radiance_hdr": "hdr",
    "targa": "tga",
    "targa_raw": "tga",
    "tiff": "tif",
    "avi_jpeg": "avi",
    "avi_raw": "avi",
    "ffmpeg": "mp4",
    "cineon": "cin",
    "dpx": "dpx",
}

PRESET_OBJECTS = {
    "STUDIO_DOME": "StudioCaptureDome",
}

CameraModel = collections.namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
Image = collections.namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name"])
Point3D = collections.namedtuple("Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
}
CAMERA_MODEL_IDS = {model.model_id: model for model in CAMERA_MODELS}
CAMERA_MODEL_NAMES = {model.model_name: model for model in CAMERA_MODELS}



# Utility helpers


def abs_render_path(scene):
    return bpy.path.abspath(scene.my_tool.render_path)



def ensure_directory(path):
    os.makedirs(path, exist_ok=True)
    return path



def get_file_extension(scene):
    file_format = scene.render.image_settings.file_format.lower()
    return FORMAT_TO_EXTENSION.get(file_format)



def list_existing_images(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(
        f for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in VALID_IMAGE_EXTENSIONS
    )



def sanitize_name(name):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)



def ensure_camera_collection(scene):
    if CAMERA_ARRAY_COLLECTION not in bpy.data.collections:
        collection = bpy.data.collections.new(CAMERA_ARRAY_COLLECTION)
        scene.collection.children.link(collection)
    return bpy.data.collections[CAMERA_ARRAY_COLLECTION]



def get_camera_array_collection():
    return bpy.data.collections.get(CAMERA_ARRAY_COLLECTION)



def get_camera_array_objects():
    collection = get_camera_array_collection()
    if not collection:
        return []
    return [obj for obj in collection.objects if obj.type == 'CAMERA']



def set_wireframe_display(obj):
    obj.display_type = 'WIRE'
    obj.show_all_edges = True
    obj.hide_render = True
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.shading.type = 'WIREFRAME'



def find_base_color_texture(material):
    if not material or not material.use_nodes:
        return None

    for node in material.node_tree.nodes:
        if node.type == 'BSDF_PRINCIPLED':
            base_color_input = node.inputs['Base Color']
            if base_color_input.is_linked:
                linked_node = base_color_input.links[0].from_node
                if linked_node.type == 'TEX_IMAGE':
                    return linked_node.image
    return None



def get_material_base_color(material):
    if material and material.use_nodes:
        for node in material.node_tree.nodes:
            if node.type == 'BSDF_PRINCIPLED':
                return node.inputs['Base Color'].default_value[:3]
    return (1.0, 1.0, 1.0)



def update_focal_length(context):
    focal_length = context.scene.my_tool.focal_length
    for obj in bpy.data.objects:
        if obj.type == 'CAMERA':
            obj.data.lens = focal_length



def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2, 2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3], 2 * qvec[1] * qvec[3] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3], 1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2, 2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[1] * qvec[3] - 2 * qvec[0] * qvec[2], 2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1], 1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2]
    ])



# Camera creation and updates


def create_array_camera(face_center, normal, collection, source_obj, index, direction):
    name = f"{source_obj.name}_ArrayCam_{direction}.{str(index + 1).zfill(3)}"
    camera_data = bpy.data.cameras.new(name=name)
    camera_obj = bpy.data.objects.new(name=name, object_data=camera_data)
    camera_obj.location = face_center
    camera_obj.scale = (0.2, 0.2, 0.2)
    camera_obj.rotation_mode = 'QUATERNION'

    look_dir = normal if direction == "IN" else -normal
    camera_obj.rotation_quaternion = look_dir.to_track_quat('Z', 'Y')

    collection.objects.link(camera_obj)
    if camera_obj.name in bpy.context.scene.collection.objects:
        bpy.context.scene.collection.objects.unlink(camera_obj)



def update_cameras_for_object(obj):
    if not obj or obj.type != 'MESH':
        return

    camera_collection = get_camera_array_collection()
    if not camera_collection:
        return

    cameras_out = [cam for cam in camera_collection.objects if cam.name.startswith(f"{obj.name}_ArrayCam_OUT")]
    cameras_in = [cam for cam in camera_collection.objects if cam.name.startswith(f"{obj.name}_ArrayCam_IN")]

    for i, face in enumerate(obj.data.polygons):
        face_center = obj.matrix_world @ face.center
        normal = obj.matrix_world.to_3x3() @ face.normal

        if i < len(cameras_out):
            cam = cameras_out[i]
            cam.location = face_center
            cam.rotation_quaternion = (-normal).normalized().to_track_quat('Z', 'Y')

        if i < len(cameras_in):
            cam = cameras_in[i]
            cam.location = face_center
            cam.rotation_quaternion = normal.normalized().to_track_quat('Z', 'Y')



def handler_update(scene, depsgraph):
    for update in depsgraph.updates:
        obj = update.id
        if isinstance(obj, bpy.types.Object) and obj.type == 'MESH':
            update_cameras_for_object(obj)



def on_new_file(scene):
    if handler_update not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(handler_update)



# Rendering and export helpers


def render_single_frame_from_array(scene, camera_collection, frame_number, frame_folder):
    scene.frame_set(frame_number)
    ensure_directory(frame_folder)

    file_extension = {
        'png': '.png',
        'jpeg': '.jpg',
        'tiff': '.tiff',
        'bmp': '.bmp',
        'openexr': '.exr',
    }.get(scene.render.image_settings.file_format.lower(), '.png')

    image_filenames = []

    for camera in camera_collection.objects:
        if camera.type != 'CAMERA':
            continue

        scene.camera = camera
        image_filename = f"{sanitize_name(camera.name)}{file_extension}"
        image_path = os.path.join(frame_folder, image_filename)
        image_filenames.append(image_filename)
        scene.render.filepath = image_path

        bpy.ops.render.render(write_still=True)
        if not os.path.exists(image_path):
            raise RuntimeError(f"Failed to save rendered image: {image_path}")

    return sorted(image_filenames)



def export_cameras_txt(scene, output_folder):
    width = scene.render.resolution_x
    height = scene.render.resolution_y
    cameras_txt_path = os.path.join(output_folder, "cameras.txt")
    cameras = [obj for obj in scene.objects if obj.type == 'CAMERA']

    with open(cameras_txt_path, 'w', encoding='utf-8') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")

        for camera_id, camera in enumerate(cameras, start=1):
            focal_length = camera.data.lens
            sensor_size = camera.data.sensor_width
            fx = fy = (focal_length / sensor_size) * width
            cx = width / 2
            cy = height / 2
            f.write(f"{camera_id} OPENCV {width} {height} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f} 0 0 0 0\n")

    return cameras_txt_path



def export_images_txt(scene, output_folder, image_filenames):
    images_txt_path = os.path.join(output_folder, "images.txt")
    camera_collection = get_camera_array_collection()
    if not camera_collection:
        raise RuntimeError("Camera Array collection not found.")

    cameras = [obj for obj in camera_collection.objects if obj.type == 'CAMERA']
    rotation_matrix = mathutils.Matrix.Rotation(math.radians(90), 4, 'X')

    with open(images_txt_path, 'w', encoding='utf-8') as f:
        f.write("# Image list with two lines per image\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")

        for img_id, (camera, img_name) in enumerate(zip(cameras, image_filenames), start=1):
            camera.matrix_world = rotation_matrix @ camera.matrix_world
            bpy.context.view_layer.update()

            cam_rot_orig = mathutils.Quaternion(camera.rotation_quaternion)
            cam_rot = mathutils.Quaternion((cam_rot_orig.x, cam_rot_orig.w, cam_rot_orig.z, -cam_rot_orig.y))
            qw, qx, qy, qz = cam_rot.w, cam_rot.x, cam_rot.y, cam_rot.z

            T = mathutils.Vector(camera.location)
            tx, ty, tz = -(cam_rot.to_matrix() @ T)

            f.write(f"{img_id} {qw} {qx} {qy} {qz} {tx:.6f} {ty:.6f} {tz:.6f} {img_id} {img_name}\n")
            f.write("0.0 0.0 0\n")

    reverse_rotation_matrix = mathutils.Matrix.Rotation(-math.radians(90), 4, 'X')
    for camera in cameras:
        camera.matrix_world = reverse_rotation_matrix @ camera.matrix_world
    bpy.context.view_layer.update()

    return images_txt_path



def export_points_txt(context, output_folder):
    scene = context.scene
    density = scene.my_tool.density
    colored_points = scene.my_tool.colored_points
    selected_obj = context.view_layer.objects.active

    if not selected_obj or selected_obj.type != 'MESH':
        raise RuntimeError("Please select a mesh object first.")

    depsgraph = context.evaluated_depsgraph_get()
    evaluated_mesh = selected_obj.evaluated_get(depsgraph).to_mesh()
    if len(evaluated_mesh.vertices) == 0:
        raise RuntimeError("No vertices found in the selected object.")

    bm = bmesh.new()
    bm.from_mesh(evaluated_mesh)

    uv_layer = bm.loops.layers.uv.active
    has_uv = uv_layer is not None
    geometry_nodes_generated = len(bm.faces) == 0 and len(bm.verts) > 0

    material_slots = selected_obj.material_slots
    material_cache = {}
    material_color_cache = {}
    texture_cache = {}
    points = []
    world_matrix = selected_obj.matrix_world

    if geometry_nodes_generated:
        for vert in bm.verts:
            world_coord = world_matrix @ vert.co
            x, y, z = world_coord.x, -world_coord.z, world_coord.y
            points.append((x, y, z, 255, 255, 255, 0.0, ""))
    else:
        for face in bm.faces:
            material_index = face.material_index
            material = material_cache.get(material_index)
            if material is None:
                material = material_slots[material_index].material if material_index < len(material_slots) else None
                material_cache[material_index] = material

            if material_index not in texture_cache:
                texture_image = find_base_color_texture(material) if has_uv else None
                if texture_image:
                    texture_cache[material_index] = (list(texture_image.pixels), texture_image.size[0], texture_image.size[1])
                else:
                    texture_cache[material_index] = ([], 0, 0)

            image_pixels, image_width, image_height = texture_cache[material_index]
            base_color = material_color_cache.setdefault(material_index, get_material_base_color(material))

            for loop in face.loops:
                vert = loop.vert
                world_coord = world_matrix @ vert.co
                x, y, z = world_coord
                new_x, new_y, new_z = x, -z, y
                r, g, b = [int(c * 255) for c in base_color]

                if colored_points and image_pixels:
                    uv = loop[uv_layer].uv
                    img_x = min(max(int(uv.x * image_width), 0), image_width - 1)
                    img_y = min(max(int(uv.y * image_height), 0), image_height - 1)
                    pixel_index = (img_y * image_width + img_x) * 4
                    if pixel_index + 2 < len(image_pixels):
                        r = int(image_pixels[pixel_index] * 255)
                        g = int(image_pixels[pixel_index + 1] * 255)
                        b = int(image_pixels[pixel_index + 2] * 255)

                points.append((new_x, new_y, new_z, r, g, b, 0.0, ""))

    total_points = len(points)
    num_points_to_keep = max(1, int((density / 100.0) * total_points))
    if num_points_to_keep < total_points:
        points = random.sample(points, num_points_to_keep)

    points_txt_path = os.path.join(output_folder, "points3D.txt")
    with open(points_txt_path, 'w', encoding='utf-8') as f:
        f.write("# 3D point list with one line per point\n")
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        for point_id, (x, y, z, r, g, b, error, track_data) in enumerate(points):
            f.write(f"{point_id} {x:.6f} {y:.6f} {z:.6f} {r} {g} {b} {error:.6f} {track_data}\n")

    bm.free()
    selected_obj.evaluated_get(depsgraph).to_mesh_clear()
    return points_txt_path, len(points)



# Operators


class GeneratePresetObjectOperator(bpy.types.Operator):
    bl_idname = "object.generate_pre_made_object"
    bl_label = "Add Preset Array Mesh"
    bl_description = "Append the hemi-dome preset camera rig from the addon asset file"

    def execute(self, context):
        mytool = context.scene.my_tool
        selected_key = mytool.object_selection

        if not os.path.exists(ASSET_BLEND_PATH):
            self.report({'ERROR'}, f"Asset file not found: {ASSET_BLEND_PATH}")
            return {'CANCELLED'}

        object_name = PRESET_OBJECTS.get(selected_key)
        if not object_name:
            self.report({'ERROR'}, "Invalid preset object selection.")
            return {'CANCELLED'}

        with bpy.data.libraries.load(ASSET_BLEND_PATH, link=False) as (data_from, data_to):
            if object_name not in data_from.objects:
                self.report({'ERROR'}, f"Object '{object_name}' not found in asset file.")
                return {'CANCELLED'}
            data_to.objects = [object_name]

        new_object = data_to.objects[0] if data_to.objects else None
        if not new_object:
            self.report({'ERROR'}, f"Failed to append preset object '{object_name}'.")
            return {'CANCELLED'}

        context.collection.objects.link(new_object)
        context.view_layer.objects.active = new_object
        context.scene.my_tool.target_object = new_object
        self.report({'INFO'}, f"Added preset object: {new_object.name}")
        return {'FINISHED'}


class CreateCamerasOperator(bpy.types.Operator):
    bl_idname = "object.create_cameras_faces"
    bl_label = "Create Cameras"
    bl_description = "Create one camera per mesh face and orient it according to the selected direction"

    def execute(self, context):
        if handler_update not in bpy.app.handlers.depsgraph_update_post:
            bpy.app.handlers.depsgraph_update_post.append(handler_update)

        selected_obj = context.active_object
        if not selected_obj or selected_obj.type != 'MESH':
            self.report({'ERROR'}, "Select a mesh object first.")
            return {'CANCELLED'}

        for modifier in selected_obj.modifiers:
            bpy.ops.object.modifier_apply(modifier=modifier.name)

        context.scene.my_tool.target_object = selected_obj
        camera_collection = ensure_camera_collection(context.scene)
        placement = context.scene.my_tool.camera_placement

        for i, face in enumerate(selected_obj.data.polygons):
            face_center = selected_obj.matrix_world @ face.center
            normal = selected_obj.matrix_world.to_3x3() @ face.normal

            if placement in {'IN', 'BOTH'}:
                create_array_camera(face_center, -normal, camera_collection, selected_obj, i, "IN")
            if placement in {'OUT', 'BOTH'}:
                create_array_camera(face_center, normal, camera_collection, selected_obj, i, "OUT")

        set_wireframe_display(selected_obj)
        self.report({'INFO'}, f"Created cameras for '{selected_obj.name}'.")
        return {'FINISHED'}


class CreateCamerasAnimationOperator(bpy.types.Operator):
    bl_idname = "object.create_cameras_animation"
    bl_label = "Create Camera Animation"
    bl_description = "Create one animated camera that visits every camera in the camera array"

    @classmethod
    def poll(cls, context):
        cameras = get_camera_array_objects()
        return bool(cameras)

    def execute(self, context):
        scene = context.scene
        cameras = get_camera_array_objects()
        if not cameras:
            self.report({'ERROR'}, "No cameras found in the Camera Array collection.")
            return {'CANCELLED'}

        animated_camera = bpy.data.objects.new("Animated_Camera", bpy.data.cameras.new("Animated_Camera"))
        scene.collection.objects.link(animated_camera)
        scene.camera = animated_camera
        animated_camera.scale = (0.2, 0.2, 0.2)

        scene.frame_start = 1
        scene.frame_end = len(cameras)

        for frame, camera in enumerate(cameras, start=1):
            animated_camera.location = camera.matrix_world.translation
            animated_camera.rotation_mode = 'QUATERNION'
            animated_camera.rotation_quaternion = camera.matrix_world.to_quaternion()
            animated_camera.keyframe_insert(data_path="location", frame=frame)
            animated_camera.keyframe_insert(data_path="rotation_quaternion", frame=frame)
            animated_camera.keyframe_insert(data_path="scale", frame=frame)

        self.report({'INFO'}, f"Created animated camera with {len(cameras)} keyframes.")
        return {'FINISHED'}


class UpdateImageCountOperator(bpy.types.Operator):
    bl_idname = "object.update_image_count"
    bl_label = "Update Image Count"
    bl_description = "Count existing rendered images in the selected output folder"

    def execute(self, context):
        render_path = abs_render_path(context.scene)
        images = list_existing_images(render_path)
        context.scene.my_tool.image_count = len(images)
        for area in bpy.context.screen.areas:
            if area.type == 'PROPERTIES':
                area.tag_redraw()
        self.report({'INFO'}, f"Found {len(images)} image files.")
        return {'FINISHED'}


class RenderCamerasOperator(bpy.types.Operator):
    bl_idname = "object.render_cameras"
    bl_label = "Render Cameras"
    bl_description = "Render one still image from every camera in the Camera Array collection"

    def execute(self, context):
        camera_collection = get_camera_array_collection()
        if not camera_collection:
            self.report({'ERROR'}, "Camera Array collection not found.")
            return {'CANCELLED'}

        cameras = [cam for cam in camera_collection.objects if cam.type == 'CAMERA']
        if not cameras:
            self.report({'ERROR'}, "No cameras found in the Camera Array collection.")
            return {'CANCELLED'}

        render_path = ensure_directory(abs_render_path(context.scene))
        file_extension = get_file_extension(context.scene)
        if not file_extension:
            self.report({'ERROR'}, "Unsupported render output format.")
            return {'CANCELLED'}

        for index, camera in enumerate(cameras, start=1):
            context.scene.camera = camera
            filename = f"{camera.name}.{file_extension}"
            full_path = os.path.join(render_path, filename)
            context.scene.render.filepath = full_path
            self.report({'INFO'}, f"Rendering camera {index}/{len(cameras)}: {camera.name}")
            bpy.ops.render.render(write_still=True)
            if not os.path.exists(full_path):
                self.report({'ERROR'}, f"Failed to save: {filename}")
                return {'CANCELLED'}

        self.report({'INFO'}, "Finished rendering all camera views.")
        return {'FINISHED'}

    def invoke(self, context, event):
        self.camera_count = len(get_camera_array_objects())
        return context.window_manager.invoke_props_dialog(self, width=320)

    def draw(self, context):
        layout = self.layout
        layout.label(text=f"Ready to render {self.camera_count} cameras.")
        layout.label(text="Progress can be monitored in the output folder.")
        layout.label(text="Rendering cannot be cancelled once started.")


class ExportCamerasOperator(bpy.types.Operator):
    bl_idname = "export.cameras"
    bl_label = "Generate cameras.txt"
    bl_description = "Export COLMAP camera intrinsics from the current Blender cameras"

    def execute(self, context):
        output_folder = abs_render_path(context.scene)
        if not os.path.isdir(output_folder):
            self.report({'ERROR'}, "Render path is invalid or missing.")
            return {'CANCELLED'}

        path = export_cameras_txt(context.scene, output_folder)
        self.report({'INFO'}, f"Saved cameras.txt to {path}")
        return {'FINISHED'}


class ExportImagesOperator(bpy.types.Operator):
    bl_idname = "export.images"
    bl_label = "Generate images.txt"
    bl_description = "Export COLMAP image extrinsics using the Camera Array collection"

    def execute(self, context):
        output_folder = abs_render_path(context.scene)
        image_files = context.scene.my_tool.get("image_filenames", [])
        if not image_files:
            self.report({'ERROR'}, "No image filenames found. Render or scan the folder first.")
            return {'CANCELLED'}

        try:
            path = export_images_txt(context.scene, output_folder, image_files)
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        self.report({'INFO'}, f"Saved images.txt to {path}")
        return {'FINISHED'}


class ExportPointsOperator(bpy.types.Operator):
    bl_idname = "export.points"
    bl_label = "Generate points3D.txt"
    bl_description = "Export a COLMAP-style 3D point cloud from the active mesh"

    def execute(self, context):
        output_folder = abs_render_path(context.scene)
        if not os.path.isdir(output_folder):
            self.report({'ERROR'}, "Render path is invalid or missing.")
            return {'CANCELLED'}

        try:
            path, point_count = export_points_txt(context, output_folder)
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        self.report({'INFO'}, f"Saved points3D.txt to {path} with {point_count} points.")
        return {'FINISHED'}


class ExportCamerasAndImagesOperator(bpy.types.Operator):
    bl_idname = "export.cameras_and_images"
    bl_label = "Generate cameras.txt + images.txt"
    bl_description = "Scan the output folder for rendered images and export matching COLMAP camera files"

    def execute(self, context):
        render_path = abs_render_path(context.scene)
        if not os.path.isdir(render_path):
            self.report({'ERROR'}, "Invalid render path.")
            return {'CANCELLED'}

        image_files = list_existing_images(render_path)
        if not image_files:
            self.report({'ERROR'}, "No rendered images were found in the selected folder.")
            return {'CANCELLED'}

        context.scene.my_tool["image_filenames"] = image_files
        bpy.ops.export.cameras()
        bpy.ops.export.images()
        self.report({'INFO'}, "Generated cameras.txt and images.txt from existing renders.")
        return {'FINISHED'}


class MergeObjectsOperator(bpy.types.Operator):
    bl_idname = "object.merge_objects"
    bl_label = "Merge Selected Meshes"
    bl_description = "Duplicate and merge selected mesh objects (and mesh children) into one mesh for point export"

    def execute(self, context):
        selected_objects = context.selected_objects
        if len(selected_objects) < 2:
            self.report({'ERROR'}, "Select at least two objects to merge.")
            return {'CANCELLED'}

        all_mesh_objects = set()
        for obj in selected_objects:
            if obj.type == 'MESH':
                all_mesh_objects.add(obj)
            for child in obj.children_recursive:
                if child.type == 'MESH':
                    all_mesh_objects.add(child)

        if len(all_mesh_objects) < 2:
            self.report({'ERROR'}, "At least two mesh objects are required.")
            return {'CANCELLED'}

        bpy.ops.object.select_all(action='DESELECT')
        for obj in all_mesh_objects:
            obj.select_set(True)
        bpy.ops.object.duplicate()
        duplicated_objects = [obj for obj in context.selected_objects if obj.select_get()]

        for obj in duplicated_objects:
            if obj.type == 'MESH' and obj.data.shape_keys is not None:
                shape_keys = obj.data.shape_keys.key_blocks
                if len(shape_keys) > 1:
                    last_key = shape_keys[-1]
                    for i, vert in enumerate(obj.data.vertices):
                        vert.co = last_key.data[i].co.copy()

        for obj in duplicated_objects:
            if obj.type == 'MESH' and obj.data.shape_keys is not None:
                context.view_layer.objects.active = obj
                bpy.ops.object.shape_key_remove(all=True)

        for obj in duplicated_objects:
            context.view_layer.objects.active = obj
            for modifier in list(obj.modifiers):
                try:
                    bpy.ops.object.modifier_apply(modifier=modifier.name)
                except RuntimeError:
                    self.report({'WARNING'}, f"Could not apply modifier '{modifier.name}' on '{obj.name}'.")

        for obj in duplicated_objects:
            if obj.type == 'MESH':
                for group in obj.vertex_groups:
                    group.name = f"{obj.name}_{group.name}"

        bpy.ops.object.select_all(action='DESELECT')
        for obj in duplicated_objects:
            obj.select_set(True)
        context.view_layer.objects.active = duplicated_objects[0]

        try:
            bpy.ops.object.join()
        except RuntimeError:
            self.report({'ERROR'}, "Failed to join objects. Ensure all selected items are meshes.")
            return {'CANCELLED'}

        merged_object = context.active_object
        if not merged_object or merged_object.type != 'MESH':
            self.report({'ERROR'}, "Merge failed.")
            return {'CANCELLED'}

        merged_object.parent = None
        merged_object.name = "Merged_point3D_mesh"
        merged_object.hide_render = False
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

        self.report({'INFO'}, f"Merged {len(all_mesh_objects)} mesh objects into {merged_object.name}.")
        return {'FINISHED'}


class AddGeometryNodesPointCloudOperator(bpy.types.Operator):
    bl_idname = "object.add_geometry_nodes_point_cloud"
    bl_label = "Add Point Cloud Geometry Nodes"
    bl_description = "Append the bundled point-cloud Geometry Nodes setup to the active mesh"

    def execute(self, context):
        selected_obj = context.active_object
        geometry_nodes_name = "PointCloudGenerator"

        if not selected_obj or selected_obj.type != 'MESH':
            self.report({'ERROR'}, "Select a mesh object first.")
            return {'CANCELLED'}

        if not os.path.exists(ASSET_BLEND_PATH):
            self.report({'ERROR'}, f"Asset file not found: {ASSET_BLEND_PATH}")
            return {'CANCELLED'}

        bpy.ops.wm.append(
            filepath=os.path.join(ASSET_BLEND_PATH, "NodeTree", geometry_nodes_name),
            directory=os.path.join(ASSET_BLEND_PATH, "NodeTree"),
            filename=geometry_nodes_name,
        )

        geometry_nodes_tree = bpy.data.node_groups.get(geometry_nodes_name)
        if not geometry_nodes_tree:
            self.report({'ERROR'}, "Failed to append the Geometry Nodes setup.")
            return {'CANCELLED'}

        modifier = selected_obj.modifiers.new(name="PointCloudGenerator", type='NODES')
        modifier.node_group = geometry_nodes_tree
        self.report({'INFO'}, "Added point cloud Geometry Nodes modifier.")
        return {'FINISHED'}


class Render4DGSAnimationOperator(bpy.types.Operator):
    bl_idname = "object.render_4dgs_animation"
    bl_label = "Render 4DGS Animation"
    bl_description = "Render a multi-view image set per frame and optionally export COLMAP files for each frame"

    def execute(self, context):
        scene = context.scene
        mytool = scene.my_tool
        animated_obj = mytool.animated_object

        if mytool.include_colmap_data and not animated_obj:
            self.report({'ERROR'}, "Select an animated mesh object before exporting per-frame COLMAP data.")
            return {'CANCELLED'}

        base_render_path = abs_render_path(scene)
        if not base_render_path:
            self.report({'ERROR'}, "Render path is not set.")
            return {'CANCELLED'}
        ensure_directory(base_render_path)

        camera_collection = get_camera_array_collection()
        if not camera_collection:
            self.report({'ERROR'}, "Camera Array collection not found.")
            return {'CANCELLED'}

        file_extension = {
            'png': '.png',
            'jpeg': '.jpg',
            'tiff': '.tiff',
            'bmp': '.bmp',
            'openexr': '.exr',
        }.get(scene.render.image_settings.file_format.lower(), '.png')

        for frame_number in range(scene.frame_start, scene.frame_end + 1, scene.frame_step):
            frame_folder = os.path.join(base_render_path, f"Frame{frame_number:04d}")

            if mytool.resume_rendering and os.path.exists(frame_folder):
                existing_files = os.listdir(frame_folder)
                if any(f.endswith(file_extension) for f in existing_files):
                    print(f"Skipping frame {frame_number}: found existing renders in {frame_folder}")
                    continue

            image_filenames = render_single_frame_from_array(scene, camera_collection, frame_number, frame_folder)

            if mytool.include_colmap_data:
                original_render_path = scene.my_tool.render_path
                scene.my_tool.render_path = frame_folder
                scene.my_tool["image_filenames"] = image_filenames
                bpy.ops.export.cameras()
                bpy.ops.export.images()

                bpy.ops.object.select_all(action='DESELECT')
                animated_obj.select_set(True)
                scene.view_layers[0].objects.active = animated_obj
                bpy.ops.export.points()
                scene.my_tool.render_path = original_render_path

        self.report({'INFO'}, "Finished rendering the 4DGS frame sequence.")
        return {'FINISHED'}

    def invoke(self, context, event):
        camera_collection = get_camera_array_collection()
        self.camera_count = len([obj for obj in camera_collection.objects if obj.type == 'CAMERA']) if camera_collection else 0
        scene = context.scene
        mytool = scene.my_tool

        self.frame_start = scene.frame_start
        self.frame_end = scene.frame_end
        self.frame_step = scene.frame_step
        self.frame_count = len(range(self.frame_start, self.frame_end + 1, self.frame_step))

        already_done = 0
        if mytool.resume_rendering:
            base_render_path = abs_render_path(scene)
            file_extension = {
                'png': '.png',
                'jpeg': '.jpg',
                'tiff': '.tiff',
                'bmp': '.bmp',
                'openexr': '.exr',
            }.get(scene.render.image_settings.file_format.lower(), '.png')
            for frame_number in range(self.frame_start, self.frame_end + 1, self.frame_step):
                frame_folder = os.path.join(base_render_path, f"Frame{frame_number:04d}")
                if os.path.exists(frame_folder) and any(f.endswith(file_extension) for f in os.listdir(frame_folder)):
                    already_done += 1

        self.already_rendered = already_done
        self.frames_to_render = self.frame_count - already_done if mytool.resume_rendering else self.frame_count
        self.resume_enabled = mytool.resume_rendering
        return context.window_manager.invoke_props_dialog(self, width=420)

    def draw(self, context):
        layout = self.layout
        if self.resume_enabled:
            layout.label(text=f"Resume mode: {self.already_rendered} frames already rendered.")
            layout.label(text=f"Remaining: {self.frames_to_render} frames x {self.camera_count} cameras = {self.frames_to_render * self.camera_count} renders.")
        else:
            layout.label(text=f"Full render: {self.frame_count} frames x {self.camera_count} cameras = {self.frame_count * self.camera_count} renders.")
        layout.label(text="Progress can be monitored in the output folder.")
        layout.label(text="Rendering cannot be cancelled once started.")



# Properties


class MyProperties(bpy.types.PropertyGroup):
    render_path: bpy.props.StringProperty(
        name="Render Path",
        description="Folder used for rendered images and COLMAP exports",
        default="//",
        subtype='DIR_PATH',
        update=lambda self, context: bpy.ops.object.update_image_count(),
    )
    image_count: bpy.props.IntProperty(
        name="Existing Images",
        description="Number of image files already found in the output folder",
        default=0,
    )
    camera_placement: bpy.props.EnumProperty(
        name="Camera Direction",
        description="Choose whether the generated cameras point inward, outward, or both",
        items=[
            ('IN', "Inward", "Cameras point inward"),
            ('OUT', "Outward", "Cameras point outward"),
            ('BOTH', "Both", "Create inward and outward cameras"),
        ],
        default='IN',
    )
    focal_length: bpy.props.FloatProperty(
        name="Focal Length",
        description="Lens focal length applied to all cameras",
        default=35.0,
        min=1.0,
        max=300.0,
        update=lambda self, context: update_focal_length(context),
    )
    target_object: bpy.props.PointerProperty(
        name="Target Object",
        description="Mesh used to drive camera-array updates",
        type=bpy.types.Object,
    )
    show_advanced: bpy.props.BoolProperty(
        name="COLMAP Export",
        description="Show or hide COLMAP export options",
        default=False,
    )
    object_selection: bpy.props.EnumProperty(
    name="Preset Array Mesh",
    description="Choose a bundled helper mesh for camera placement",
    items=[
        ('STUDIO_DOME', "Studio Capture Dome", "Half-dome multi-view rig for human capture"),
    ],
    default='STUDIO_DOME',
)
    density: bpy.props.IntProperty(
        name="Point Density",
        description="Percentage of exported points to keep",
        default=100,
        min=1,
        max=100,
    )
    colored_points: bpy.props.BoolProperty(
        name="Use Vertex / Texture Colors",
        description="Export colored points when material data is available",
        default=True,
    )
    animated_object: bpy.props.PointerProperty(
        name="Animated Object",
        description="Animated mesh used for per-frame 4DGS exports",
        type=bpy.types.Object,
    )
    show_animated_4dgs: bpy.props.BoolProperty(
        name="Animated 4DGS Data",
        description="Show or hide per-frame rendering tools",
        default=False,
    )
    include_colmap_data: bpy.props.BoolProperty(
        name="Export COLMAP Data",
        description="Generate cameras.txt, images.txt, and points3D.txt for each frame",
        default=True,
    )
    show_additional_tools: bpy.props.BoolProperty(
        name="Additional Tools",
        description="Show or hide advanced utility tools",
        default=False,
    )
    resume_rendering: bpy.props.BoolProperty(
        name="Resume Rendering",
        description="Skip frames that already contain rendered images",
        default=True,
    )


# UI


class CameraArrayPanel(bpy.types.Panel):
    bl_label = f"4DGS Data Tool v{'.'.join(map(str, bl_info['version']))}"
    bl_idname = "VIEW3D_PT_camera_array_tool"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "4DGS Tool"

    def draw(self, context):
        layout = self.layout
        mytool = context.scene.my_tool

        box = layout.box()
        box.label(text="Studio Camera Rig", icon='CUBE')
        box.prop(mytool, "object_selection")
        box.operator("object.generate_pre_made_object", text="Add Preset Mesh")

        box = layout.box()
        box.label(text="Camera Array", icon='CAMERA_DATA')
        box.prop(mytool, "camera_placement")
        box.prop(mytool, "target_object")
        box.operator("object.create_cameras_faces")
        box.operator("object.create_cameras_animation", text="Create Animated Camera")
        box.prop(mytool, "focal_length")

        box = layout.box()
        box.label(text="Rendering", icon='RENDER_STILL')
        box.prop(mytool, "render_path")
        box.label(text=f"Existing images: {mytool.image_count}", icon='IMAGE_DATA')
        box.operator("object.render_cameras")

        layout.prop(mytool, "show_advanced", icon="TRIA_DOWN" if mytool.show_advanced else "TRIA_RIGHT", emboss=True)
        if mytool.show_advanced:
            colmap_box = layout.box()
            colmap_box.label(text="COLMAP Export", icon='EXPORT')
            colmap_box.operator("export.cameras_and_images", text="Generate cameras.txt + images.txt")
            colmap_box.prop(mytool, "density")
            colmap_box.prop(mytool, "colored_points")
            colmap_box.operator("export.points", text="Generate points3D.txt")

            colmap_box.prop(mytool, "show_additional_tools", icon="TRIA_DOWN" if mytool.show_additional_tools else "TRIA_RIGHT", emboss=True)
            if mytool.show_additional_tools:
                tools_box = colmap_box.box()
                tools_box.label(text="Utilities", icon='TOOL_SETTINGS')
                tools_box.operator("object.merge_objects", text="Merge Selected Meshes")
                tools_box.operator("object.add_geometry_nodes_point_cloud", text="Add Point Cloud Nodes")

        layout.prop(mytool, "show_animated_4dgs", icon="TRIA_DOWN" if mytool.show_animated_4dgs else "TRIA_RIGHT", emboss=True)
        if mytool.show_animated_4dgs:
            anim_box = layout.box()
            anim_box.label(text="Animated 4DGS Export", icon='ARMATURE_DATA')
            anim_box.prop(mytool, "animated_object")
            anim_box.prop(mytool, "include_colmap_data")
            anim_box.prop(mytool, "resume_rendering")
            anim_box.operator("object.render_4dgs_animation", text="Render 4DGS Sequence")



# Registration


CLASSES = (
    MyProperties,
    GeneratePresetObjectOperator,
    CreateCamerasOperator,
    CreateCamerasAnimationOperator,
    UpdateImageCountOperator,
    RenderCamerasOperator,
    ExportCamerasOperator,
    ExportImagesOperator,
    ExportPointsOperator,
    ExportCamerasAndImagesOperator,
    MergeObjectsOperator,
    AddGeometryNodesPointCloudOperator,
    Render4DGSAnimationOperator,
    CameraArrayPanel,
)



def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)

    bpy.types.Scene.my_tool = bpy.props.PointerProperty(type=MyProperties)

    if on_new_file not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(on_new_file)
    if handler_update not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(handler_update)



def unregister():
    if handler_update in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(handler_update)
    if on_new_file in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(on_new_file)

    if hasattr(bpy.types.Scene, "my_tool"):
        del bpy.types.Scene.my_tool

    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
