"""Blender scene for the Crystal Caves dinosaur (run inside Blender, headless).

    blender --background --factory-startup --python ArtSource/Blender/dinosaur_scene.py -- OUT_DIR

One Blender unit is one final sprite pixel. The creature faces +X and stands on
z = 1 so the ink outline fits under its feet in a 24x64 frame. Every frame is
rendered twice: a flat pass whose object colors identify each material exactly,
and a studio-lit pass whose brightness becomes the pixel-art shading ramp.
``scripts/render_blender_sprites.py`` turns both passes into palette pixels.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import bpy
from mathutils import Euler, Vector

SCALE = 4  # supersampling per final pixel
WIDTH, HEIGHT = 24, 64

# Flat ID colors; the post-process maps each to a palette ramp.
BODY = (0.10, 0.80, 0.10)
FAR = (0.10, 0.40, 0.10)
BELLY = (0.90, 0.90, 0.10)
WHITE = (1.0, 1.0, 1.0)
PUPIL = (0.0, 0.0, 1.0)
CLAW = (1.0, 0.5, 0.0)

# Four walk poses: (near foot x, far foot x, body bob).
POSES = ((-3.0, 2.5, 0.0), (1.0, -0.5, 1.0), (3.0, -3.0, 0.0), (-2.0, 1.5, 1.0))


def blob(name, location, radii, color, rotation=(0.0, 0.0, 0.0)):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, location=location)
    item = bpy.context.active_object
    item.name = name
    item.scale = radii
    item.rotation_euler = Euler([math.radians(angle) for angle in rotation])
    item.color = (*color, 1.0)
    bpy.ops.object.shade_smooth()
    return item


def limb(name, start, end, radius, color):
    start, end = Vector(start), Vector(end)
    middle = (start + end) / 2
    direction = end - start
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=16, radius=radius, depth=direction.length, location=middle
    )
    item = bpy.context.active_object
    item.name = name
    item.rotation_euler = direction.to_track_quat("Z", "Y").to_euler()
    item.color = (*color, 1.0)
    bpy.ops.object.shade_smooth()
    for point in (start, end):
        blob(name + " joint", point, (radius, radius, radius), color)
    return item


def build(pose):
    near_foot, far_foot, bob = pose
    for item in list(bpy.data.objects):
        bpy.data.objects.remove(item, do_unlink=True)
    up = Vector((0, 0, bob))
    # Legs first: the far leg sits behind the body in a darker ID color.
    for side, foot_x, color, depth in (
        ("far", far_foot, FAR, 3.0),
        ("near", near_foot, BODY, -3.0),
    ):
        hip = Vector((-1.5, depth, 20.0)) + up
        knee = Vector(((foot_x - 1.5) / 2 + 0.5, depth, 10.0)) + up * 0.5
        ankle = Vector((foot_x, depth, 3.0))
        blob(side + " thigh", hip, (4.2, 3.0, 6.5), color, (0, -15, 0))
        limb(side + " shin", knee, ankle, 1.6, color)
        limb(side + " upper shin", hip, knee, 2.4, color)
        blob(side + " foot", (foot_x + 1.8, depth, 2.2), (3.4, 2.0, 1.2), color)
        for toe in (foot_x + 4.2, foot_x + 5.0):
            blob(side + " toe claw", (toe, depth - 1.6, 1.6), (0.6, 0.6, 0.6), CLAW)
    # Tail curls down behind the hips.
    for index in range(9):
        t = index / 8
        x, z = -5.5 - 5.0 * math.sin(t * 1.4), 23.0 - 13.0 * t
        radius = 4.0 - 2.9 * t
        blob(f"tail {index}", Vector((x, 0.5, z)) + up, (radius, radius, radius), BODY)
    blob("torso", Vector((-0.5, 0.0, 31.0)) + up, (6.0, 5.5, 12.0), BODY, (0, -12, 0))
    blob("belly", Vector((1.8, -1.5, 29.0)) + up, (3.8, 4.5, 9.5), BELLY, (0, -12, 0))
    blob("neck", Vector((2.0, 0.0, 43.0)) + up, (4.0, 4.0, 7.5), BODY, (0, -20, 0))
    blob("skull", Vector((2.5, 0.0, 53.0)) + up, (5.5, 4.5, 5.0), BODY)
    blob("upper jaw", Vector((7.0, 0.0, 51.6)) + up, (5.0, 3.6, 2.4), BODY, (0, 6, 0))
    blob("lower jaw", Vector((6.0, 0.0, 46.4)) + up, (4.6, 3.2, 1.4), BODY, (0, -8, 0))
    blob("chin", Vector((4.0, -0.5, 46.2)) + up, (3.0, 3.0, 1.2), BELLY, (0, -8, 0))
    # A dark open mouth separates the jaws; sparse teeth hang into it.
    blob("mouth", Vector((7.0, -2.0, 48.6)) + up, (4.6, 1.5, 1.0), PUPIL, (0, -2, 0))
    for x in (5.5, 8.5):
        blob("upper tooth", Vector((x, -3.4, 49.2)) + up, (0.45, 0.45, 0.7), WHITE)
    blob("lower tooth", Vector((7.0, -3.4, 48.0)) + up, (0.45, 0.45, 0.6), WHITE)
    blob("brow", Vector((3.2, -2.4, 56.0)) + up, (2.6, 2.0, 0.9), FAR, (0, 12, 0))
    blob("eye", Vector((3.6, -3.6, 54.2)) + up, (1.3, 1.0, 1.3), WHITE)
    blob("pupil", Vector((4.2, -4.4, 54.1)) + up, (0.6, 0.6, 0.8), PUPIL)
    blob("nostril", Vector((10.8, -2.2, 52.6)) + up, (0.5, 0.5, 0.5), PUPIL)
    # Short clawed arms, held forward.
    limb("arm", Vector((4.0, -4.0, 37.0)) + up, Vector((8.0, -4.5, 34.5)) + up, 1.1, BODY)
    blob("hand claw", Vector((9.0, -4.8, 34.0)) + up, (0.8, 0.8, 0.8), CLAW)


def camera_and_render(directory, frame):
    scene = bpy.context.scene
    bpy.ops.object.camera_add(location=(0, -100, HEIGHT / 2), rotation=(math.radians(90), 0, 0))
    camera = bpy.context.active_object
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = HEIGHT
    scene.camera = camera
    render = scene.render
    render.engine = "BLENDER_WORKBENCH"
    render.resolution_x, render.resolution_y = WIDTH * SCALE, HEIGHT * SCALE
    render.resolution_percentage = 100
    render.film_transparent = True
    render.image_settings.file_format = "PNG"
    render.image_settings.color_mode = "RGBA"
    scene.display.render_aa = "OFF"
    shading = scene.display.shading
    shading.color_type = "OBJECT"
    shading.show_cavity = False
    shading.show_object_outline = False
    shading.show_specular_highlight = False
    scene.view_settings.view_transform = "Standard"
    for name, light in (("id", "FLAT"), ("shade", "STUDIO")):
        shading.light = light
        render.filepath = str(directory / f"dinosaur_{frame}_{name}.png")
        bpy.ops.render.render(write_still=True)


def main():
    directory = Path(sys.argv[sys.argv.index("--") + 1])
    directory.mkdir(parents=True, exist_ok=True)
    for frame, pose in enumerate(POSES):
        build(pose)
        camera_and_render(directory, frame)


main()
