"""Blender models for every moving Crystal Caves creature (run inside Blender).

    blender --background --factory-startup --python ArtSource/Blender/creatures.py -- OUT [NAME ...]

One Blender unit is one final sprite pixel. Creatures are authored in sprite
pixel coordinates through ``at(col, row, depth)`` (row 0 is the top of the
sprite, negative depth is toward the camera), so a feature such as an eye can
be placed on the exact pixel the game's art tests expect. Each pose is
rendered as a flat pass whose colors identify slot and layer (see
``sprite_ids.py``) and a studio-lit pass of neutral gray whose brightness
becomes the shading, once for the native pixel sprite and once for the HD
sprite. Parts made with ``detail()`` (catchlights, scales, cracks, gloss) only
appear in HD, where they are large enough to read.
``scripts/render_blender_sprites.py`` converts the passes.
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

SCALE = 8  # render pixels per sprite pixel; HD sprites are half of this (4x)
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
    joints = (blob(start, (radius,) * 3, slot, layer), blob(end, (end_radius,) * 3, slot, layer))
    item["joints"] = [joint.name for joint in joints]
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


def tube(points, radii, slot="body", layer=BODY):
    """A smooth tapered tube through world points (tails, snake bodies)."""
    curve = bpy.data.curves.new("tube", "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = 1.0
    curve.bevel_resolution = 8
    curve.resolution_u = 16
    curve.use_fill_caps = True
    # Bezier with automatic handles passes through every point.
    spline = curve.splines.new("BEZIER")
    spline.bezier_points.add(len(points) - 1)
    for point, location, radius in zip(spline.bezier_points, points, radii):
        point.co = location
        point.radius = radius
        point.handle_left_type = point.handle_right_type = "AUTO"
    spline.use_smooth = True
    item = bpy.data.objects.new("tube", curve)
    bpy.context.collection.objects.link(item)
    tag(item, slot, layer)
    # Round end caps.
    blob(points[0], (radii[0],) * 3, slot, layer)
    blob(points[-1], (radii[-1],) * 3, slot, layer)
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


def detail(item):
    """Mark a part as HD-only: too fine for a native-resolution pixel."""
    item["hd_only"] = True
    for name in item.get("joints", ()):
        bpy.data.objects[name]["hd_only"] = True
    return item


def catchlight(location, size=0.3):
    return detail(blob(location, (size, size * 0.6, size), "white", NEAR))


# --- creatures ------------------------------------------------------------


def spike(base, tip, radius, slot="dark", layer=BODY):
    """A pointed cone: back ridges, claws, teeth and ears."""
    return limb(base, tip, radius, slot, layer, 0.04)


def dinosaur(frame: int) -> None:
    """Tyrannosaurus, facing right, feet on the bottom row.

    A huge boxy head with a deep jaw on a short thick neck, a forward-leaning
    body on massive thighs, tiny two-fingered arms, and a heavy tail that
    sweeps back down to the ground as a counterweight. One continuous spine
    carries tail, body and neck so the silhouette reads as a single animal.
    """
    near_x, near_lift, far_x, far_lift, bob = (
        (-3.5, 0.0, 4.5, 0.0, 0.0),
        (0.5, 0.0, 1.0, 3.0, 1.0),
        (4.5, 0.0, -3.5, 0.0, 0.0),
        (1.0, 3.0, 0.5, 0.0, 1.0),
    )[frame]
    up = Vector((0, 0, bob))
    sway = (0.0, 0.6, 0.0, -0.6)[frame]
    nod = Vector((0, 0, (0.0, -0.5, 0.0, 0.5)[frame]))
    # (x, z, radius) from the tail tip to the back of the skull.
    spine = [
        (-11.0 + sway, 3.2, 0.8),
        (-9.8 + sway * 0.8, 8.5, 2.3),
        (-7.6 + sway * 0.4, 14.5, 4.0),
        (-4.0, 21.5, 6.0),
        (-0.4, 28.0, 6.6),
        (2.0, 33.5, 5.9),
        (2.2, 38.5, 5.0),
        (1.6, 43.0, 4.6),
    ]
    points = [
        Vector((x, 0.5, z)) + up + nod * (i / (len(spine) - 1)) ** 2
        for i, (x, z, _) in enumerate(spine)
    ]
    tube(points, [r for _, _, r in spine])
    # Pale belly up the front of the chest and throat into the lower jaw.
    belly = [
        (0.2, 21.0, 2.8),
        (3.0, 28.5, 4.0),
        (4.4, 35.0, 3.4),
        (4.2, 40.5, 2.6),
        (5.0, 44.5, 2.0),
    ]
    tube([Vector((x, -3.0, z)) + up for x, z, _ in belly], [r for _, _, r in belly], "belly")
    for foot_x, lift, depth, layer in (
        (far_x, far_lift, 3.4, FAR),
        (near_x, near_lift, -3.4, NEAR),
    ):
        # Massive drumstick thigh, long shin, raised ankle, three clawed toes.
        hip = Vector((-1.6, depth, 21.0)) + up
        ankle = Vector((foot_x - 0.8, depth, 5.4 + lift))
        knee = Vector(((hip.x + foot_x) / 2 + 2.8, depth, 12.5 + lift / 2)) + up * 0.5
        # The near thigh is part of the body mass, so no contour rings it.
        thigh_layer = BODY if layer == NEAR else FAR
        blob(hip + Vector((0.6, 0, -1.0)), (6.4, 3.8, 8.0), "body", thigh_layer, (0, -20, 0))
        limb(hip, knee, 3.6, "body", layer, 2.7)
        limb(knee, ankle, 2.7, "body", layer, 2.0)
        ball = Vector((foot_x + 1.6, depth, 1.9 + lift))
        limb(ankle, ball, 2.0, "body", layer, 1.6)
        for spread, reach in ((-1.2, 2.5), (0.0, 3.1), (1.2, 2.5)):
            toe = ball + Vector((reach, spread - 0.6, -0.5))
            limb(ball, toe, 1.1, "body", layer, 0.8)
            spike(toe, toe + Vector((1.3, -0.3, -0.5)), 0.6, "accent", layer)
    # Head: a long, deep, boxy skull with a heavy lower jaw, mouth agape.
    head = up + nod
    scale = 1.18

    def hp(x: float, y: float, z: float) -> Vector:
        # Head parts are authored around the jaw hinge, then enlarged.
        return Vector(((x - 2.4) * scale + 1.6, y * scale, (z - 50.0) * scale + 49.2)) + head

    def hr(x: float, y: float, z: float) -> tuple[float, float, float]:
        return (x * scale, y * scale, z * scale)

    blob(hp(2.4, 0.0, 52.6), hr(4.4, 4.0, 4.4), "body")  # cranium
    blob(hp(2.2, 0.0, 48.0), hr(3.8, 3.8, 3.4), "body")  # jaw muscle
    blob(hp(6.6, 0.0, 52.6), hr(4.8, 3.4, 2.8), "body", BODY, (0, 4, 0))  # snout
    blob(hp(9.6, 0.0, 52.2), hr(1.5, 2.8, 2.3), "body")  # blunt nose
    blob(hp(6.2, 0.0, 47.6), hr(4.6, 3.0, 1.7), "body", BODY, (0, -4, 0))  # lower jaw
    blob(hp(5.8, -1.2, 47.0), hr(4.0, 2.4, 1.0), "belly", BODY, (0, -4, 0))
    blob(hp(7.0, -1.6, 49.7), hr(4.2, 1.6, 0.9), "ink", BODY, (0, 2, 0))  # mouth
    for x, keep in ((4.2, False), (5.6, True), (7.0, False), (8.4, True), (9.6, False)):
        tooth = hp(x, -2.6, 50.6 + (x - 4.2) * 0.07)
        part = spike(tooth, tooth + Vector((0.1, 0, -1.3)), 0.45, "white")
        if not keep:
            detail(part)
    for x, keep in ((5.0, False), (6.4, True), (7.8, False), (9.0, True)):
        tooth = hp(x, -2.3, 48.9 - (x - 5.0) * 0.05)
        part = spike(tooth, tooth + Vector((0, 0, 1.1)), 0.4, "white")
        if not keep:
            detail(part)
    # Small fierce eye under a heavy brow ridge, set back on the skull.
    blob(hp(3.6, -3.4, 54.0), hr(1.1, 0.9, 1.1), "white")
    blob(hp(4.1, -4.1, 53.9), hr(0.45, 0.5, 0.8), "ink")
    blob(hp(3.8, -3.2, 55.6), hr(2.2, 1.3, 0.75), "dark", BODY, (0, 18, 0))
    catchlight(hp(4.3, -4.5, 54.3), 0.24)
    blob(hp(10.4, -1.4, 53.6), hr(0.5, 0.5, 0.4), "ink")  # nostril
    detail(limb(hp(0.6, -3.4, 50.8), hp(3.4, -3.6, 49.4), 0.3, "dark"))
    # A row of small scutes down the back.
    for index in range(2, 7):
        (x0, z0, _), (x1, z1, _) = spine[index - 1], spine[index + 1]
        x, z, r = spine[index]
        tangent = Vector((x1 - x0, 0, z1 - z0)).normalized()
        back = Vector((-tangent.z, 0, tangent.x))
        base = Vector((x, 0.5, z)) + up + back * r * 0.9
        detail(spike(base, base + back * 1.0 + tangent * 0.4, 0.7, "dark", BODY))
    # Tiny arms with two clawed fingers.
    swing = (1.0, 0.0, -1.0, 0.0)[frame]
    shoulder = Vector((6.0, -4.6, 35.5)) + up
    elbow = Vector((7.6, -5.0, 33.4 + swing * 0.3)) + up
    hand = Vector((9.2 + swing * 0.4, -5.1, 34.0 + swing * 0.6)) + up
    limb(shoulder, elbow, 1.3, "body", NEAR, 1.05)
    limb(elbow, hand, 1.05, "body", NEAR, 0.8)
    for spread in (-0.5, 0.5):
        finger = hand + Vector((0.6, 0.0, spread - 0.3))
        spike(finger, finger + Vector((0.8, -0.2, -0.5)), 0.45, "accent", NEAR)


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
        limb(Vector((shoulder.x, 0.8, shoulder.z)), wrist_world, 0.7, "belly", FAR, 0.55)
        for tip in tips:
            x, z = place(*tip)
            limb(wrist_world, Vector((x, 0.6, z)), 0.55, "belly", FAR, 0.3)
        # Thumb claw at the wrist.
        detail(
            spike(wrist_world, wrist_world + Vector((side * 0.4, -0.4, 0.9)), 0.35, "white", FAR)
        )
    blob(center, (2.7, 2.4, 4.0), "dark")
    blob(center + Vector((0, -0.8, -0.6)), (1.5, 2.0, 2.6), "body")
    # Tiny clawed feet tucked under the body.
    for side in (-1, 1):
        foot = center + Vector((side * 1.0, -0.6, -3.8))
        limb(foot, foot + Vector((side * 0.2, 0, -1.2)), 0.45, "belly", BODY, 0.3)
    head = center + Vector((0, -0.4, 4.6))
    blob(head, (2.5, 2.2, 2.2), "dark")
    for side in (-1, 1):
        # Tall pointed ears with pink inner skin.
        ear = head + Vector((side * 1.3, 0, 1.4))
        spike(ear, ear + Vector((side * 1.2, 0, 3.0)), 0.95, "dark", BODY)
        inner = ear + Vector((side * 0.1, -0.6, 0.3))
        detail(spike(inner, inner + Vector((side * 0.9, -0.1, 2.2)), 0.5, "tongue", BODY))
        blob(head + Vector((side * 1.25, -2.0, 0.3)), (0.5, 0.4, 0.5), "eye")
        blob(head + Vector((side * 0.55, -2.1, -1.5)), (0.3, 0.3, 0.6), "white")
        catchlight(head + Vector((side * 1.1, -2.4, 0.5)), 0.17)
    detail(blob(head + Vector((0, -2.3, -0.4)), (0.55, 0.3, 0.35), "tongue", NEAR))  # nose
    for index in range(7):
        offset = Vector(((index - 3) * 0.55, -2.3, -0.8 - abs(index - 3) * 0.35))
        detail(blob(center + offset, (0.4, 0.3, 0.9), "dark", NEAR))  # chest fur
    for index in range(5):
        detail(
            spike(
                head + Vector(((index - 2) * 0.5, -0.6, 1.8)),
                head + Vector(((index - 2) * 0.8, -0.8, 2.8)),
                0.3,
                "dark",
                BODY,
            )
        )  # head tuft


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

    tube(
        [at(col + sway(index), row) for index, (col, row, _) in enumerate(path)],
        [radius for _, _, radius in path],
    )
    # A smooth pale belly stripe along the front of the raised neck.
    belly = path[4:10]
    tube(
        [at(c + sway(i + 4) + r * 0.6, row, -r * 0.5) for i, (c, row, r) in enumerate(belly)],
        [r * 0.42 for _, _, r in belly],
        "belly",
    )
    # Dark diamond markings down the back.
    for index in range(2, 10):
        col, row, radius = path[index]
        mark = at(col + sway(index) - radius * 0.25, row, -radius * 0.82)
        detail(blob(mark, (radius * 0.42, 0.3, radius * 0.42), "dark", BODY, (0, 45, 0)))
    # Wedge-shaped viper head: wide cheeks behind a narrower snout.
    blob(at(18.6, 3.4), (3.6, 2.4, 2.2), "body", BODY, (0, -8, 0))
    blob(at(17.2, 4.3), (2.4, 2.7, 1.8), "body", BODY, (0, -15, 0))
    blob(at(19.0, 2.0, -1.9), (0.7, 0.5, 0.7), "eye")
    blob(at(19.4, 2.0, -2.3), (0.22, 0.25, 0.5), "ink")
    catchlight(at(18.8, 1.7, -2.4), 0.16)
    detail(blob(at(18.6, 1.2, -1.2), (1.6, 1.0, 0.5), "body", NEAR, (0, -10, 0)))  # brow
    detail(blob(at(21.6, 3.0, -1.6), (0.25, 0.25, 0.25), "ink"))  # nostril
    detail(limb(at(16.4, 4.9, -2.3), at(21.8, 4.5, -1.4), 0.12, "ink"))  # jaw line
    if frame in (0, 2):
        tip = 23.2 if frame == 2 else 22.6
        base = at(21.6, 4.6, -0.3)
        limb(base, at(tip, 5.0, -0.3), 0.32, "tongue", NEAR, 0.25)
        for fork in (-0.5, 0.5):
            detail(spike(at(tip, 5.0, -0.3), at(tip + 0.8, 5.0 + fork, -0.3), 0.2, "tongue", NEAR))


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
        catchlight(at(19.7, row - 0.2, -6.2), 0.25)
    # Wet gloss streak, a small grin and suspended bubbles inside the jelly.
    detail(blob(at(11.5, 7.2 - stretch, -4.2), (2.6, 0.4, 0.7), "white", NEAR, (0, 14, 0)))
    detail(blob(at(21.3, 12.6, -4.6), (0.5, 0.3, 1.1), "ink", NEAR, (0, 0, 0)))
    for col, row, size in ((10.0, 14.5, 0.7), (13.5, 16.5, 0.5), (8.0, 11.0, 0.45)):
        detail(blob(at(col, row, -4.0), (size,) * 3, "belly", BODY))
    # Two hanging drips that stretch and recoil as it bobs.
    for col in ((12.0, 13.0, 11.0, 12.0)[frame], 19.0):
        top = at(col + 0.5, 18.6 + stretch, 1.0)
        drop = at(col + 0.7, 20.8 + stretch, 1.0)
        limb(top, drop, 1.4, "dark", FAR, 1.05)
        blob(drop + Vector((0, 0, -0.5)), (1.3, 1.2, 1.4), "dark", FAR)


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
            for toe in (-0.7, 0.4, 1.5):
                detail(
                    blob(foot + Vector((1.9, toe - 0.6, -0.8)), (0.6, 0.55, 0.5), "belly", layer)
                )
    body = boulder(at(11.5, 11.0 + bob), (10.6, 7.0, 8.6), seed=7)
    body.rotation_euler = Euler((0.0, math.radians(8), 0.2))
    # Moss tufts on the crown.
    for col, row, size in ((8.0, 4.4, 3.2), (13.5, 3.8, 2.6), (17.5, 5.6, 1.8)):
        blob(at(col, row + bob, -2.0), (size, 2.2, 1.1), "accent", BODY, (0, 0, 0), True)
        for index in range(4):
            offset = (index - 1.5) * size * 0.5
            tuft = at(col + offset, row + bob - 0.8 - (index % 2) * 0.4, -2.6)
            detail(blob(tuft, (size * 0.32, 1.0, size * 0.32), "accent", BODY))
    if not asleep:
        # A heavy stone brow over watchful eyes.
        detail(boulder(at(17.3, 9.1 + bob, -6.4), (3.4, 0.9, 0.8), seed=3, layer=NEAR))
        for col in (15.5, 19.0):
            blob(at(col, 11.0 + bob, -6.6), (1.4, 0.6, 1.3), "white")
            blob(at(col + 0.5, 11.4 + bob, -7.0), (0.75, 0.4, 0.8), "ink")
            catchlight(at(col + 0.2, 10.8 + bob, -7.4), 0.22)
    # Cracks across the facets, mineral flecks and loose pebbles on the crown.
    for points in (((4.0, 9.0), (6.5, 12.5), (5.5, 16.0)), ((11.0, 15.0), (13.0, 17.5))):
        for (c1, r1), (c2, r2) in zip(points, points[1:]):
            detail(limb(at(c1, r1 + bob, -7.0), at(c2, r2 + bob, -7.0), 0.18, "ink", NEAR))
    for col, row in ((7.5, 6.5), (3.5, 12.0), (9.5, 16.5)):
        detail(boulder(at(col, row + bob, -7.2), (0.35, 0.3, 0.3), seed=int(col * 7), slot="white"))
    for col, row in ((10.5, 3.6), (15.8, 4.6)):
        detail(boulder(at(col, row + bob, -2.6), (0.8, 0.8, 0.6), seed=int(col * 10), layer=NEAR))


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
    for light in list(bpy.data.lights):
        bpy.data.lights.remove(light)


GLOSSY = {"slug_enemy", "eye_flyer"}


def material(name: str, roughness: float, coat: float = 0.0):
    item = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    item.use_nodes = True
    shader = item.node_tree.nodes["Principled BSDF"]
    shader.inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1.0)
    shader.inputs["Roughness"].default_value = roughness
    shader.inputs["Coat Weight"].default_value = coat
    return item


def light_for_hd(scene, objects, glossy: bool) -> None:
    """Path-traced key, rim and sky light for the HD shading pass.

    The key comes from the upper left in front, a strong rim from behind on
    the right outlines the silhouette, and soft sky fill keeps shadow sides
    readable. Only brightness is used, so materials stay neutral gray.
    """
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 96
    scene.cycles.use_denoising = True
    scene.cycles.seed = 1  # deterministic noise pattern
    world = scene.world or bpy.data.worlds.new("sky")
    scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes["Background"]
    background.inputs["Color"].default_value = (0.32, 0.34, 0.42, 1.0)
    background.inputs["Strength"].default_value = 0.6
    for label, toward, strength in (
        ("key", Vector((0.55, 0.75, -0.85)), 3.2),
        ("rim", Vector((-0.75, -0.9, -0.35)), 4.0),
    ):
        data = bpy.data.lights.new(label, "SUN")
        data.energy = strength
        data.angle = math.radians(12)
        lamp = bpy.data.objects.new(label, data)
        lamp.rotation_euler = toward.to_track_quat("-Z", "Y").to_euler()
        scene.collection.objects.link(lamp)
    clay = material("clay", 0.55)
    gloss = material("gloss", 0.2, 0.6)
    for item in objects:
        shiny = glossy or item["slot"] in ("white", "eye", "ink")
        item.data.materials.clear()
        item.data.materials.append(gloss if shiny else clay)


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
    for prefix, hd in (("", False), ("hd_", True)):
        for item in ids:
            item.hide_render = item.get("hd_only", False) and not hd
        for pass_name in ("id", "shade"):
            if pass_name == "id":
                shading.light = "FLAT"
                shading.show_specular_highlight = shading.show_cavity = False
                for item, color in ids.items():
                    item.color = color
            else:
                # Neutral gray: brightness alone drives the tone ramp.
                for item in ids:
                    item.color = (0.8, 0.8, 0.8, 1.0)
                shading.light = "STUDIO"
                shading.show_specular_highlight = True
            if hd and pass_name == "shade":
                light_for_hd(scene, ids, glossy=name in GLOSSY)
            else:
                settings.engine = "BLENDER_WORKBENCH"
            settings.filepath = str(directory / f"{name}_{frame}_{prefix}{pass_name}.png")
            bpy.ops.render.render(write_still=True)


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
