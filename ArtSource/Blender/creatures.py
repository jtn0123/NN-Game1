"""Blender models for every moving Crystal Caves creature (run inside Blender).

    blender --background --factory-startup --python ArtSource/Blender/creatures.py -- OUT [NAME ...]

One Blender unit is one final sprite pixel. Creatures are authored in sprite
pixel coordinates through ``at(col, row, depth)`` (row 0 is the top of the
sprite, negative depth is toward the camera), so a feature such as an eye can
be placed on the exact pixel the game's art tests expect. Each pose is
rendered twice: a flat pass whose colors identify slot and layer (see
``sprite_ids.py``) and a studio-lit pass of neutral gray whose brightness
becomes the pixel shading. ``scripts/render_blender_sprites.py`` converts both.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import bmesh
import bpy
from mathutils import Euler, Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sprite_ids import BODY, FAR, NEAR, id_color  # noqa: E402

SCALE = 4  # supersampling per final pixel
SIZE = [24, 24]  # current sprite (width, height); set per creature


def at(col: float, row: float, depth: float = 0.0) -> Vector:
    """World position of a sprite pixel center."""
    return Vector((col - SIZE[0] / 2 + 0.5, depth, SIZE[1] - row - 0.5))


def tag(item, slot: str, layer: int) -> None:
    item["slot"], item["layer"] = slot, layer
    item.color = (*id_color(slot, layer), 1.0)


def blob(location, radii, slot="body", layer=BODY, rotation=(0.0, 0.0, 0.0), flat=False):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, location=location)
    item = bpy.context.active_object
    item.scale = radii
    item.rotation_euler = Euler([math.radians(angle) for angle in rotation])
    tag(item, slot, layer)
    if flat:
        bpy.ops.object.shade_flat()
    else:
        bpy.ops.object.shade_smooth()
    return item


def limb(start, end, radius, slot="body", layer=BODY, end_radius=None):
    """A tapered capsule between two points."""
    start, end = Vector(start), Vector(end)
    direction = end - start
    end_radius = radius if end_radius is None else end_radius
    bpy.ops.mesh.primitive_cone_add(
        vertices=16,
        radius1=radius,
        radius2=end_radius,
        depth=max(direction.length, 1e-3),
        location=(start + end) / 2,
    )
    item = bpy.context.active_object
    item.rotation_euler = direction.to_track_quat("Z", "Y").to_euler()
    tag(item, slot, layer)
    bpy.ops.object.shade_smooth()
    blob(start, (radius,) * 3, slot, layer)
    blob(end, (end_radius,) * 3, slot, layer)
    return item


def membrane(points, depth, thickness, bulge, slot="body", layer=FAR):
    """A billowing sheet through 2D (x, z) world points, e.g. a bat wing."""
    mesh = bpy.data.meshes.new("membrane")
    built = bmesh.new()
    corners = [Vector((x, depth, z)) for x, z in points]
    center = sum(corners, Vector()) / len(corners)
    reach = max((corner - center).length for corner in corners)
    built.faces.new([built.verts.new(corner) for corner in corners])
    bmesh.ops.triangulate(built, faces=built.faces[:])
    bmesh.ops.subdivide_edges(built, edges=built.edges[:], cuts=3, use_grid_fill=True)
    for vert in built.verts:
        falloff = max(0.0, 1 - ((vert.co - center).length / reach) ** 2)
        vert.co.y += bulge * falloff
    extruded = bmesh.ops.extrude_face_region(built, geom=built.faces[:])
    moved = [element for element in extruded["geom"] if isinstance(element, bmesh.types.BMVert)]
    bmesh.ops.translate(built, verts=moved, vec=(0, thickness, 0))
    bmesh.ops.recalc_face_normals(built, faces=built.faces[:])
    built.to_mesh(mesh)
    built.free()
    item = bpy.data.objects.new("membrane", mesh)
    bpy.context.collection.objects.link(item)
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    tag(item, slot, layer)
    return item


def boulder(location, radii, seed, slot="body", layer=BODY):
    """A faceted, deterministic rock: a jittered icosphere with flat shading."""
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=1.0, location=location)
    item = bpy.context.active_object
    rng = random.Random(seed)
    for vert in item.data.vertices:
        vert.co *= 1 + rng.uniform(-0.13, 0.13)
    item.scale = radii
    tag(item, slot, layer)
    bpy.ops.object.shade_flat()
    return item


# --- creatures ------------------------------------------------------------


def dinosaur(frame: int) -> None:
    """Upright green dinosaur, facing right, feet on the bottom row."""
    near_x, near_lift, far_x, far_lift, bob = (
        (-4.5, 0.0, 4.0, 0.0, 0.0),
        (-0.5, 0.0, 0.5, 3.0, 1.0),
        (4.0, 0.0, -4.5, 0.0, 0.0),
        (0.5, 3.0, -0.5, 0.0, 1.0),
    )[frame]
    up = Vector((0, 0, bob))
    sway = (0.0, 0.6, 0.0, -0.6)[frame]
    for foot_x, lift, depth, layer in (
        (far_x, far_lift, 3.0, FAR),
        (near_x, near_lift, -3.0, NEAR),
    ):
        hip = Vector((-1.5, depth, 21.0)) + up
        ankle = Vector((foot_x, depth, 3.6 + lift))
        knee = Vector(((hip.x + foot_x) / 2 + 2.2, depth, 11.5 + lift / 2)) + up * 0.5
        blob(hip, (5.0, 3.4, 7.2), "body", layer, (0, -15, 0))
        limb(hip, knee, 3.0, "body", layer, 2.4)
        limb(knee, ankle, 2.3, "body", layer, 1.8)
        blob((foot_x + 1.6, depth, 2.0 + lift), (3.6, 2.2, 1.5), "body", layer)
        for toe in (foot_x + 4.0, foot_x + 4.9):
            blob((toe, depth - 1.8, 1.4 + lift), (0.7, 0.7, 0.7), "accent", layer)
    # Thick tail sweeping down behind the hips.
    for index in range(9):
        t = index / 8
        x = -5.0 - 4.6 * math.sin(t * 1.4) + sway * t
        blob(Vector((x, 0.5, 24.0 - 13.5 * t)) + up, (4.6 - 3.3 * t,) * 3, "body")
    blob(Vector((-0.5, 0.0, 31.0)) + up, (7.0, 6.0, 12.5), "body", BODY, (0, -12, 0))
    blob(Vector((2.2, -2.0, 29.0)) + up, (4.2, 4.8, 9.8), "belly", BODY, (0, -12, 0))
    blob(Vector((2.4, 0.0, 43.5)) + up, (4.6, 4.4, 7.5), "body", BODY, (0, -20, 0))
    head = up + Vector((0, 0, (0.0, -0.4, 0.0, 0.4)[frame]))
    blob(Vector((2.6, 0.0, 53.0)) + head, (5.8, 4.8, 5.2), "body")
    blob(Vector((6.4, 0.0, 51.8)) + head, (4.6, 3.8, 2.6), "body", BODY, (0, 6, 0))
    blob(Vector((5.6, 0.0, 46.6)) + head, (4.4, 3.4, 1.5), "body", BODY, (0, -8, 0))
    blob(Vector((3.8, -0.6, 46.3)) + head, (3.0, 3.2, 1.3), "belly", BODY, (0, -8, 0))
    blob(Vector((6.6, -2.2, 48.8)) + head, (4.2, 1.6, 1.1), "ink", BODY, (0, -2, 0))
    for x in (5.2, 8.2):
        blob(Vector((x, -3.6, 49.4)) + head, (0.5, 0.5, 0.75), "white")
    blob(Vector((6.7, -3.6, 48.1)) + head, (0.5, 0.5, 0.65), "white")
    blob(Vector((3.2, -2.6, 56.4)) + head, (2.8, 2.2, 1.0), "dark", BODY, (0, 12, 0))
    blob(Vector((3.6, -3.8, 54.3)) + head, (1.4, 1.0, 1.4), "white")
    blob(Vector((4.3, -4.6, 54.2)) + head, (0.65, 0.6, 0.85), "ink")
    blob(Vector((10.0, -2.4, 52.6)) + head, (0.55, 0.55, 0.55), "ink")
    swing = (1.0, 0.0, -1.0, 0.0)[frame]
    shoulder = Vector((4.0, -4.2, 37.5)) + up
    hand = Vector((8.0 + swing * 0.6, -4.6, 34.5 + swing)) + up
    limb(shoulder, hand, 1.5, "body", NEAR, 1.2)
    blob(hand + Vector((1.0, -0.2, -0.4)), (0.85, 0.85, 0.85), "accent", NEAR)


def bat(frame: int) -> None:
    """Dusky purple-black bat seen from the front, wings flapping."""
    angle = math.radians((50.0, 18.0, -14.0, -40.0)[frame])
    center = at(11.5, 13.5 + (0.0, 0.4, 0.8, 0.4)[frame])
    for side in (-1, 1):
        shoulder = center + Vector((side * 1.6, 0, 1.6))

        def place(x, z, side=side, shoulder=shoulder):
            # Template points are for the right wing; rotate about the shoulder.
            rx = x * math.cos(angle) - z * math.sin(angle)
            rz = x * math.sin(angle) + z * math.cos(angle)
            return shoulder.x + side * rx, shoulder.z + rz

        wrist = (4.4, 2.0)
        tips = ((9.8, 3.6), (10.0, -1.6), (6.6, -5.0))
        # Deep scallops between the finger tips give the classic bat edge.
        outline = [(0.0, 0.0), wrist, tips[0], (6.9, 1.0), tips[1], (6.0, -2.0), tips[2]]
        outline += [(3.2, -2.8), (0.8, -4.6)]
        points = [place(x, z) for x, z in outline]
        if side < 0:
            points.reverse()
        membrane(points, 1.6, 0.6, -1.2, "body", FAR)
        wrist_world = Vector((place(*wrist)[0], 0.8, place(*wrist)[1]))
        limb(
            Vector((shoulder.x, 0.8, shoulder.z)),
            wrist_world,
            0.7,
            "belly",
            FAR,
            0.55,
        )
        for tip in tips:
            x, z = place(*tip)
            limb(wrist_world, Vector((x, 0.6, z)), 0.55, "belly", FAR, 0.3)
    blob(center, (2.7, 2.4, 4.0), "dark")
    blob(center + Vector((0, -0.8, -0.6)), (1.5, 2.0, 2.6), "body")
    head = center + Vector((0, -0.4, 4.6))
    blob(head, (2.5, 2.2, 2.2), "dark")
    for side in (-1, 1):
        blob(
            head + Vector((side * 1.4, 0, 2.3)), (0.75, 0.6, 1.5), "dark", BODY, (0, side * -18, 0)
        )
        blob(head + Vector((side * 1.25, -2.0, 0.3)), (0.5, 0.4, 0.5), "eye")
        blob(head + Vector((side * 0.55, -2.1, -1.5)), (0.3, 0.3, 0.6), "white")


def slug(frame: int) -> None:
    """Upright pink snake with a floor-hugging tail, facing right."""
    phase = frame * math.pi / 2
    path = [
        (1.0, 28.5, 0.7),
        (3.5, 29.4, 1.3),
        (7.0, 29.0, 2.0),
        (10.0, 28.0, 2.6),
        (12.4, 25.8, 3.0),
        (13.6, 22.6, 3.0),
        (13.0, 19.0, 2.8),
        (12.8, 15.0, 2.6),
        (13.6, 11.0, 2.4),
        (15.0, 7.6, 2.2),
        (16.6, 5.0, 2.0),
    ]

    def sway(index: float) -> float:
        return 1.3 * math.sin(phase + index * 0.7) if 4 <= index <= 8 else 0.0

    for index, (col, row, radius) in enumerate(path):
        blob(at(col + sway(index), row), (radius, radius * 0.9, radius))
    # A smooth pale belly stripe along the front of the raised neck.
    for step in range(4 * 4, 9 * 4 + 1):
        t = step / 4
        low, high = path[int(t)], path[min(len(path) - 1, int(t) + 1)]
        mix = t - int(t)
        col, row, radius = (a + (b - a) * mix for a, b in zip(low, high))
        blob(at(col + sway(t) + radius * 0.6, row, -radius * 0.5), (radius * 0.42,) * 3, "belly")
    blob(at(18.6, 3.4), (3.6, 2.4, 2.2), "body", BODY, (0, -8, 0))
    blob(at(19.0, 2.0, -1.9), (0.7, 0.5, 0.7), "eye")
    blob(at(19.4, 2.0, -2.3), (0.25, 0.25, 0.4), "ink")
    if frame in (0, 2):
        tip = 23.2 if frame == 2 else 22.6
        limb(at(21.6, 4.6, -0.3), at(tip, 5.0, -0.3), 0.32, "tongue", NEAR, 0.25)


def eye_flyer(frame: int) -> None:
    """Glossy green flying slime: tapered tail at the left, two stacked eyes."""
    stretch = (0.0, 0.5, 0.0, -0.5)[frame]
    wag = (0.0, -0.6, 0.0, 0.6)[frame]
    blob(at(15.0, 12.5), (7.2, 5.0, 8.0 + stretch), "body")
    blob(at(9.0, 12.4), (4.4, 4.2, 4.6 + stretch * 0.5), "body")
    blob(at(4.6, 12.2 + wag * 0.5), (2.6, 2.4, 1.9), "body")
    blob(at(2.4, 12.0 + wag), (1.2, 1.1, 1.0), "body")
    blob(at(13.0, 6.5 - stretch, -2.5), (4.0, 2.6, 1.4), "belly")
    for row in (9, 16):
        blob(at(19.0, row, -4.0), (2.2, 1.0, 2.2), "dark")
        blob(at(19.0, row, -5.0), (1.6, 0.5, 1.6), "white")
        blob(at(20.0, row + 0.4, -5.6), (0.6, 0.4, 1.0), "ink")
    for col in ((12.0, 13.0, 11.0, 12.0)[frame], 19.0):
        blob(at(col + 0.5, 20.5 + stretch, 1.0), (1.3, 1.2, 1.5), "dark", FAR)


def walking_rock(frame: int) -> None:
    """Faceted boulder on stubby purple legs; frame 4 is the sleeping rock."""
    asleep = frame == 4
    bob = 3.0 if asleep else (0.0, 1.0, 0.0, 1.0)[frame]
    if not asleep:
        near, far = ((7.0, 16.5), (9.5, 14.0), (12.0, 11.5), (9.5, 14.0))[frame]
        lift_near, lift_far = ((0, 0), (1.2, 0), (0, 0), (0, 1.2))[frame]
        for col, lift, depth, layer in ((far, lift_far, 2.5, FAR), (near, lift_near, -2.5, NEAR)):
            hip = at(col, 17.0, depth)
            foot = at(col, 21.6 - lift, depth)
            limb(hip, foot, 1.5, "belly", layer, 1.3)
            blob(foot + Vector((0.8, 0, -0.6)), (2.0, 1.4, 0.9), "belly", layer)
    body = boulder(at(11.5, 11.0 + bob), (10.6, 7.0, 8.6), seed=7)
    body.rotation_euler = Euler((0.0, math.radians(8), 0.2))
    # Moss on the crown and a few darker strata.
    for col, row, size in ((8.0, 4.4, 3.2), (13.5, 3.8, 2.6), (17.5, 5.6, 1.8)):
        blob(at(col, row + bob, -2.0), (size, 2.2, 1.1), "accent", BODY, (0, 0, 0), True)
    if not asleep:
        for col in (15.5, 19.0):
            blob(at(col, 11.0 + bob, -6.6), (1.4, 0.6, 1.3), "white")
            blob(at(col + 0.5, 11.4 + bob, -7.0), (0.75, 0.4, 0.8), "ink")


CREATURES = {
    "dinosaur_enemy": ((24, 64), 4, dinosaur),
    "bat_enemy": ((24, 24), 4, bat),
    "slug_enemy": ((24, 32), 4, slug),
    "eye_flyer": ((24, 24), 4, eye_flyer),
    "walking_rock": ((24, 24), 5, walking_rock),
}


def clear() -> None:
    for item in list(bpy.data.objects):
        bpy.data.objects.remove(item, do_unlink=True)
    for mesh in list(bpy.data.meshes):
        bpy.data.meshes.remove(mesh)


def render(directory: Path, name: str, frame: int) -> None:
    scene = bpy.context.scene
    width, height = SIZE
    bpy.ops.object.camera_add(location=(0, -100, height / 2), rotation=(math.radians(90), 0, 0))
    camera = bpy.context.active_object
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = max(width, height)
    scene.camera = camera
    settings = scene.render
    settings.engine = "BLENDER_WORKBENCH"
    settings.resolution_x, settings.resolution_y = width * SCALE, height * SCALE
    settings.resolution_percentage = 100
    settings.film_transparent = True
    settings.image_settings.file_format = "PNG"
    settings.image_settings.color_mode = "RGBA"
    scene.display.render_aa = "OFF"
    scene.view_settings.view_transform = "Standard"
    shading = scene.display.shading
    shading.color_type = "OBJECT"
    shading.show_object_outline = False
    shading.show_cavity = False
    ids = {item: tuple(item.color) for item in scene.objects if "slot" in item}
    for pass_name in ("id", "shade"):
        if pass_name == "id":
            shading.light = "FLAT"
            shading.show_specular_highlight = False
        else:
            # Neutral gray, studio lit: brightness alone drives the tone ramp.
            shading.light = "STUDIO"
            shading.show_specular_highlight = True
            for item in ids:
                item.color = (0.8, 0.8, 0.8, 1.0)
        settings.filepath = str(directory / f"{name}_{frame}_{pass_name}.png")
        bpy.ops.render.render(write_still=True)
    for item, color in ids.items():
        item.color = color


def main() -> None:
    arguments = sys.argv[sys.argv.index("--") + 1 :]
    directory = Path(arguments[0])
    directory.mkdir(parents=True, exist_ok=True)
    for name in arguments[1:] or list(CREATURES):
        size, frames, build = CREATURES[name]
        SIZE[:] = size
        for frame in range(frames):
            clear()
            build(frame)
            render(directory, name, frame)


main()
