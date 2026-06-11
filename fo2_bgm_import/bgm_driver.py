# FlatOut Blender Tools — https://github.com/RavenDS/flatout-blender-tools
# bgm_driver.py — FlatOut 2 driver skeleton import.
# Parses <name>_bones.ini, builds a Blender armature, assigns vertex groups by
# OBB segment testing, and parents the mesh objects to the armature.
# Coordinate convention matches the car importer: FO2 (x,y,z) → Blender (x,z,y).

import re
import os
import bpy
from mathutils import Vector, Matrix, Quaternion
from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BoneData:
    """Bone entry from the Bones = { } section of a bones.ini file."""
    name:         str   = ""
    index:        int   = 0
    position:     tuple = (0.0, 0.0, 0.0)        # FO2 model-space position
    orientation:  tuple = (0.0, 0.0, 0.0, 1.0)   # quaternion (x,y,z,w)
    model_to_bone: list = field(default_factory=list)  # 4×4 row-major matrix


@dataclass
class SegmentData:
    """Bounding-box entry from the Segments = { } section of a bones.ini file."""
    name:        str   = ""
    bone_index:  int   = 0
    dimension:   tuple = (0.0, 0.0, 0.0)         # full extent (not half)
    position:    tuple = (0.0, 0.0, 0.0)          # OBB centre in FO2 model space
    orientation: tuple = (0.0, 0.0, 0.0, 1.0)    # quaternion (x,y,z,w)


# ─────────────────────────────────────────────────────────────────────────────
# STANDARD FO2 DRIVER SKELETON HIERARCHY
# key = bone name, value = parent bone name (None = root)
# ─────────────────────────────────────────────────────────────────────────────

SKELETON_HIERARCHY = {
    'head':         None,           # root
    'upper_torso':  'head',
    'lower_torso':  'upper_torso',
    'upper_arm_l':  'upper_torso',
    'lower_arm_l':  'upper_arm_l',
    'upper_arm_r':  'upper_torso',
    'lower_arm_r':  'upper_arm_r',
    'upper_leg_l':  'lower_torso',
    'lower_leg_l':  'upper_leg_l',
    'upper_leg_r':  'lower_torso',
    'lower_leg_r':  'upper_leg_r',
}


# ─────────────────────────────────────────────────────────────────────────────
# BONES.INI PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_bones_ini(filepath):
    """Parse a FlatOut 2 driver bones INI file.

    Returns (bones_dict, segments_dict):
      bones_dict    — {int_index: BoneData}
      segments_dict — {bone_name_str: SegmentData}
    Returns ({}, {}) on failure.
    """
    try:
        with open(filepath, 'r', errors='replace') as f:
            text = f.read()
    except (OSError, IOError) as exc:
        print(f"[Driver] ERROR: Cannot read bones INI '{filepath}': {exc}")
        return {}, {}

    # ── Bones ──
    # Pattern matches:  -- bone_name\n\t[N] = {\n\t\t...fields...\n\t}
    bones = {}
    bone_pat = re.compile(
        r'--\s*(\w+)\s*\n\s*\[(\d+)\]\s*=\s*\{(.*?)\n\t\}',
        re.DOTALL,
    )
    for m in bone_pat.finditer(text):
        name  = m.group(1)
        idx   = int(m.group(2))
        block = m.group(3)

        bone = BoneData(name=name, index=idx)

        pos_m = re.search(r'Position\s*=\s*\{([^}]+)\}', block)
        if pos_m:
            bone.position = tuple(float(x) for x in pos_m.group(1).split(','))

        ori_m = re.search(r'Orientation\s*=\s*\{([^}]+)\}', block)
        if ori_m:
            bone.orientation = tuple(float(x) for x in ori_m.group(1).split(','))

        mtb_m = re.search(r'ModelToBone\s*=\s*\{(.*?)\n\t\t\}', block, re.DOTALL)
        if mtb_m:
            rows = []
            for rm in re.finditer(r'\[\d+\]\s*=\s*\{([^}]+)\}', mtb_m.group(1)):
                rows.append([float(x) for x in rm.group(1).split(',')])
            if len(rows) == 4:
                bone.model_to_bone = rows

        bones[idx] = bone

    if not bones:
        print(f"[Driver] WARNING: No bones found in '{filepath}'")
        return {}, {}

    # ── Segments ──
    # Parse the outermost "Segments = { ... \n}" block (not the indexed sub-blocks).
    segments = {}
    seg_m = re.search(r'\nSegments\s*=\s*\{(.*?)\n\}', text, re.DOTALL)
    if seg_m:
        seg_text = seg_m.group(1)
        seg_pat = re.compile(
            r'(\w+)\s*=\s*\{\s*\n'
            r'\s*BoneIndex\s*=\s*(\d+),\s*\n'
            r'\s*Dimension\s*=\s*\{([^}]+)\},\s*\n'
            r'\s*Position\s*=\s*\{([^}]+)\},\s*\n'
            r'\s*Orientation\s*=\s*\{([^}]+)\}',
        )
        for sm in seg_pat.finditer(seg_text):
            name = sm.group(1)
            segments[name] = SegmentData(
                name=name,
                bone_index=int(sm.group(2)),
                dimension=tuple(float(x) for x in sm.group(3).split(',')),
                position=tuple(float(x) for x in sm.group(4).split(',')),
                orientation=tuple(float(x) for x in sm.group(5).split(',')),
            )

    print(f"[Driver] Parsed {len(bones)} bones, {len(segments)} segments "
          f"from '{os.path.basename(filepath)}'")
    return bones, segments


def parse_eject_poses(filepath):
    """Parse the EjectPoseSegments block from a bones INI.

    Structure:
        EjectPoseSegments = {
            [1] = { name = { BoneIndex, Dimension, Position, Orientation }, ... },
            [2] = { ... },
            ...
        }
    Each pose reuses the same segment (OBB) boxes as the rest Segments block but
    repositioned/reoriented into the driver-eject configuration.

    Returns a list (one entry per pose) of dicts:
        { bone_name: (position_tuple, orientation_tuple) }
    Returns [] when the section is absent or unparsable.
    """
    try:
        with open(filepath, 'r', errors='replace') as f:
            text = f.read()
    except (OSError, IOError):
        return []

    start = text.find('EjectPoseSegments')
    if start < 0:
        return []
    block = text[start:]

    # Same per-segment fields as the Segments block, at any indentation.
    seg_pat = re.compile(
        r'(\w+)\s*=\s*\{\s*'
        r'BoneIndex\s*=\s*(\d+)\s*,\s*'
        r'Dimension\s*=\s*\{[^}]+\}\s*,\s*'
        r'Position\s*=\s*\{([^}]+)\}\s*,\s*'
        r'Orientation\s*=\s*\{([^}]+)\}',
        re.DOTALL,
    )
    # Pose boundaries are the top-level [N] = { markers inside this block.
    pose_starts = [m.start() for m in re.finditer(r'\[\d+\]\s*=\s*\{', block)]
    if not pose_starts:
        return []
    pose_starts.append(len(block))

    poses = []
    for i in range(len(pose_starts) - 1):
        seg_text = block[pose_starts[i]:pose_starts[i + 1]]
        pose = {}
        for sm in seg_pat.finditer(seg_text):
            pose[sm.group(1)] = (
                tuple(float(x) for x in sm.group(3).split(',')),
                tuple(float(x) for x in sm.group(4).split(',')),
            )
        if pose:
            poses.append(pose)
    return poses


# ─────────────────────────────────────────────────────────────────────────────
# COORDINATE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _fo2_to_bl(fo2_pos, scale=1.0):
    """FO2 (x,y,z) → Blender (x,z,y) — matches the car-import axis convention."""
    x, y, z = fo2_pos
    return Vector((x * scale, z * scale, y * scale))


def _fo2_quat_to_bl(q):
    """FO2 orientation quaternion (x,y,z,w) → Blender mathutils Quaternion.

    The FO2→Blender axis swap (x,y,z)→(x,z,y) is a Y/Z reflection (det = −1).
    Conjugating a rotation by a reflection swaps the Y/Z axis components and
    negates the rotation, which works out to (w, −x, −z, −y) in mathutils
    (w,x,y,z) order. Verified numerically against S·R·S over random rotations.
    """
    x, y, z, w = q
    return Quaternion((w, -x, -z, -y))


def _box_matrix(position, orientation, scale=1.0):
    """Build a 4×4 armature-space transform for an OBB (segment) box."""
    loc = _fo2_to_bl(position, scale)
    rot = _fo2_quat_to_bl(orientation).to_matrix().to_4x4()
    return Matrix.Translation(loc) @ rot


def _quat_to_matrix_3x3(q):
    """Quaternion (x,y,z,w) → 3×3 rotation matrix (row-major list-of-lists)."""
    x, y, z, w = q
    return [
        [1 - 2*(y*y + z*z),  2*(x*y - z*w),      2*(x*z + y*w)     ],
        [2*(x*y + z*w),      1 - 2*(x*x + z*z),   2*(y*z - x*w)    ],
        [2*(x*z - y*w),      2*(y*z + x*w),        1 - 2*(x*x + y*y)],
    ]


def _point_in_obb_ratio(pt, center, half_ext, rot_mat):
    """Return the max normalised penetration depth of point inside the OBB.
    0.0 = at centre, 1.0 = exactly on the surface, >1.0 = outside.
    """
    dx = pt[0] - center[0]
    dy = pt[1] - center[1]
    dz = pt[2] - center[2]
    max_ratio = 0.0
    for axis in range(3):
        proj = (dx * rot_mat[axis][0] +
                dy * rot_mat[axis][1] +
                dz * rot_mat[axis][2])
        if half_ext[axis] > 1e-9:
            ratio = abs(proj) / half_ext[axis]
            if ratio > max_ratio:
                max_ratio = ratio
        elif abs(proj) > 1e-6:
            return 9999.0
    return max_ratio


# ─────────────────────────────────────────────────────────────────────────────
# ARMATURE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_armature(context, bones, segments, hierarchy,
                   global_scale=1.0, armature_name="DriverSkeleton"):
    """Create a Blender armature from FO2 bone data.

    Bone head positions use the same FO2→Blender axis swap as the car importer
    so that the armature aligns with a driver BGM imported through the car plugin.

    Returns the armature Object.
    """
    arm_data = bpy.data.armatures.new(armature_name)
    arm_obj  = bpy.data.objects.new(armature_name, arm_data)
    context.collection.objects.link(arm_obj)
    context.view_layer.objects.active = arm_obj

    bone_by_name = {b.name: b for b in bones.values()}

    bpy.ops.object.mode_set(mode='EDIT')
    edit_bones = arm_data.edit_bones
    created    = {}

    for bone_name, parent_name in hierarchy.items():
        if bone_name not in bone_by_name:
            print(f"[Driver] WARNING: '{bone_name}' not found in INI data, skipping")
            continue

        bd   = bone_by_name[bone_name]
        eb   = edit_bones.new(bone_name)
        head = _fo2_to_bl(bd.position, global_scale)
        eb.head = head

        # Tail: point toward the average position of direct children; fall back
        # to extending in the parent→self direction for leaf bones.
        children = [cn for cn, pn in hierarchy.items()
                    if pn == bone_name and cn in bone_by_name]

        if children:
            avg = sum(
                (_fo2_to_bl(bone_by_name[cn].position, global_scale) for cn in children),
                Vector((0.0, 0.0, 0.0)),
            ) / len(children)
            direction = avg - head
            if direction.length < 0.01:
                direction = Vector((0.0, 0.05, 0.0))
            eb.tail = head + direction
        else:
            if parent_name and parent_name in bone_by_name:
                parent_pos = _fo2_to_bl(bone_by_name[parent_name].position, global_scale)
                direction  = head - parent_pos
                if direction.length < 0.01:
                    direction = Vector((0.0, 0.05, 0.0))
                else:
                    direction = direction.normalized() * min(direction.length * 0.5, 0.15)
            else:
                direction = Vector((0.0, 0.05, 0.0))
            eb.tail = head + direction

        created[bone_name] = eb

    # Wire up parent relationships. Bones are intentionally left UNCONNECTED
    # (use_connect stays False)
    for bone_name, parent_name in hierarchy.items():
        if parent_name and bone_name in created and parent_name in created:
            created[bone_name].parent = created[parent_name]

    bpy.ops.object.mode_set(mode='OBJECT')
    return arm_obj


# ─────────────────────────────────────────────────────────────────────────────
# VERTEX → BONE ASSIGNMENT
# ─────────────────────────────────────────────────────────────────────────────

def _assign_vertices_to_bones(mesh_objects, segments, global_scale=1.0):
    """Assign every vertex of every mesh object to its nearest bone segment (OBB test).

    Segments are defined in FO2 model space.  The car importer transforms
    vertices as FO2(x,y,z) → Blender(x,z,y)×scale, so we reverse that:
        FO2(x,y,z) = Blender(x,z,y) / scale

    Returns {bone_name: {mesh_obj: [vertex_indices]}}
    """
    # Pre-compute OBB data for each segment once
    seg_obbs = []
    for seg in segments.values():
        seg_obbs.append({
            'name':     seg.name,
            'center':   seg.position,
            'half_ext': (seg.dimension[0] / 2.0,
                         seg.dimension[1] / 2.0,
                         seg.dimension[2] / 2.0),
            'rot_mat':  _quat_to_matrix_3x3(seg.orientation),
        })

    if not seg_obbs:
        return {}

    inv_scale = 1.0 / global_scale if global_scale else 1.0
    result    = {}  # bone_name → {obj → [vert_indices]}

    for obj in mesh_objects:
        if obj.type != 'MESH':
            continue
        mw = obj.matrix_world

        for vi, vert in enumerate(obj.data.vertices):
            # World-space Blender position
            wbl = mw @ vert.co
            # Inverse car-import transform: Blender(x,y,z) → FO2(x,z,y) / scale
            fo2_pt = (wbl.x * inv_scale, wbl.z * inv_scale, wbl.y * inv_scale)

            best_name = None
            best_dist = 9999.0
            for seg in seg_obbs:
                d = _point_in_obb_ratio(
                    fo2_pt, seg['center'], seg['half_ext'], seg['rot_mat'])
                if d < best_dist:
                    best_dist = d
                    best_name = seg['name']

            if best_name is not None:
                result.setdefault(best_name, {}).setdefault(obj, []).append(vi)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# EJECT POSES
# ─────────────────────────────────────────────────────────────────────────────

def _import_eject_poses(context, arm_obj, segments, eject_poses, global_scale=1.0):
    """Import the EjectPoseSegments as keyframed armature poses.

    Each eject pose gives, per segment, the OBB box's position + orientation.
    A segment box is rigidly attached to its bone, so the box's rest→eject world
    motion equals the bone's: D = T_eject_box · T_rest_box⁻¹, and the bone's posed
    armature-space matrix is D · bone.matrix_local. That matrix is converted to a
    pose-bone basis and keyframed.

    Frame 1 holds the rest pose; frames 2..N+1 hold the eject poses. Timeline
    markers name each frame. Returns the number of eject poses imported.
    """
    pose_bones = arm_obj.pose.bones

    # rest-pose box transform (armature space) for each segment we can pose
    rest_box = {}
    for name, seg in segments.items():
        if pose_bones.get(name) is not None:
            rest_box[name] = _box_matrix(seg.position, seg.orientation, global_scale)
    if not rest_box:
        return 0

    arm_obj.animation_data_create()
    action = bpy.data.actions.get("fo2_driver_eject_poses")
    if action is None:
        action = bpy.data.actions.new("fo2_driver_eject_poses")
    arm_obj.animation_data.action = action

    def _apply(pose_dict):
        # 1) armature-space posed matrix for every bone (rest where no data)
        m_eject = {}
        for pb in pose_bones:
            ml = pb.bone.matrix_local
            if pose_dict and pb.name in rest_box and pb.name in pose_dict:
                pos, ori = pose_dict[pb.name]
                t_eject = _box_matrix(pos, ori, global_scale)
                m_eject[pb.name] = (t_eject @ rest_box[pb.name].inverted()) @ ml
            else:
                m_eject[pb.name] = ml.copy()
        # 2) convert each to a pose basis (parent-relative), order-independent
        for pb in pose_bones:
            ml = pb.bone.matrix_local
            if pb.parent is not None:
                par_ml = pb.parent.bone.matrix_local
                par_e  = m_eject.get(pb.parent.name, par_ml)
                pb.matrix_basis = ml.inverted() @ par_ml @ par_e.inverted() @ m_eject[pb.name]
            else:
                pb.matrix_basis = ml.inverted() @ m_eject[pb.name]

    def _key(frame, label):
        for pb in pose_bones:
            pb.keyframe_insert('location', frame=frame)
            pb.keyframe_insert('rotation_quaternion', frame=frame)
        existing = {m.frame for m in context.scene.timeline_markers}
        if frame not in existing:
            context.scene.timeline_markers.new(label, frame=frame)

    # frame 1 = rest, frames 2.. = eject poses
    _apply(None)
    _key(1, "rest")
    for i, pose in enumerate(eject_poses):
        _apply(pose)
        _key(i + 2, f"eject_{i + 1}")

    # leave the rig displayed in its rest pose
    context.scene.frame_set(1)
    return len(eject_poses)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def import_driver_skeleton(context, mesh_objects, bones_ini_path, global_scale=1.0):
    """Parse bones_ini_path and attach a skeleton to mesh_objects.

    Steps:
      1. Parse the bones.ini file.
      2. Build a Blender armature using the standard FO2→Blender axis swap.
      3. Assign vertex groups by OBB segment testing.
      4. Parent every mesh object to the armature and add an Armature modifier.

    mesh_objects   — list of Blender mesh Objects (from build_blender_meshes)
    bones_ini_path — absolute path to the <name>_bones.ini file
    global_scale   — must match the scale used during the mesh import

    Returns the armature Object, or None on failure.
    """
    bones, segments = parse_bones_ini(bones_ini_path)
    if not bones:
        print(f"[Driver] Skeleton import aborted: no bones in '{bones_ini_path}'")
        return None

    # Build hierarchy: start from the known standard hierarchy and add any
    # extra bones found in the file as disconnected roots.
    hierarchy = dict(SKELETON_HIERARCHY)
    for bd in bones.values():
        if bd.name not in hierarchy:
            hierarchy[bd.name] = None

    # Driver naming: always "fo2_driver" / "fo2_driver_armature" regardless of
    # male/female, so the export side can capture the names without per-gender
    # special-casing. (A driver BGM contains a single model.)
    driver_mesh = next((o for o in mesh_objects if o.type == 'MESH'), None)
    if driver_mesh is not None:
        driver_mesh.name = "fo2_driver"
    arm_name = "fo2_driver_armature"

    arm_obj = build_armature(context, bones, segments, hierarchy,
                             global_scale=global_scale, armature_name=arm_name)

    # Stamp each bone's ModelToBone
    mtb_by_name = {}
    for bd in bones.values():
        if len(bd.model_to_bone) == 4:
            flat = [float(v) for row in bd.model_to_bone for v in row]
            if len(flat) == 16:
                mtb_by_name[bd.name] = flat
    if mtb_by_name:
        arm_obj["fo2_driver_mtb"] = mtb_by_name

    # build_armature links the armature to the active collection; move it into
    # the "FO2 Body" collection so it sits with the driver mesh. Leave it in
    # place if that collection somehow doesn't exist.
    fo2_body_coll = bpy.data.collections.get("FO2 Body")
    if fo2_body_coll is not None:
        for c in list(arm_obj.users_collection):
            c.objects.unlink(arm_obj)
        fo2_body_coll.objects.link(arm_obj)

    # ── Vertex group assignment ──
    if segments:
        bone_vertex_map = _assign_vertices_to_bones(
            mesh_objects, segments, global_scale)
        for bone_name, obj_map in bone_vertex_map.items():
            for obj, indices in obj_map.items():
                if not indices:
                    continue
                vg = obj.vertex_groups.get(bone_name)
                if vg is None:
                    vg = obj.vertex_groups.new(name=bone_name)
                vg.add(indices, 1.0, 'REPLACE')
    else:
        print("[Driver] WARNING: No segments in bones.ini — vertex groups not assigned")

    # ── Parent meshes to armature ──
    for obj in mesh_objects:
        if obj.type != 'MESH':
            continue
        obj.parent = arm_obj
        mod        = obj.modifiers.new(name="Armature", type='ARMATURE')
        mod.object = arm_obj

    # ── Eject poses ── (optional/additive; never fail the whole import over them)
    try:
        eject_poses = parse_eject_poses(bones_ini_path)
        if eject_poses:
            n = _import_eject_poses(context, arm_obj, segments,
                                    eject_poses, global_scale)
            print(f"[Driver] Imported {n} eject pose(s) "
                  f"(frame 1 = rest, frames 2..{n + 1} = eject poses)")
    except Exception as exc:
        print(f"[Driver] WARNING: eject pose import failed (non-fatal): {exc}")

    # Leave armature active and selected
    bpy.ops.object.select_all(action='DESELECT')
    arm_obj.select_set(True)
    for obj in mesh_objects:
        obj.select_set(True)
    context.view_layer.objects.active = arm_obj

    print(f"[Driver] Skeleton import complete: {len(bones)} bones attached to "
          f"{len([o for o in mesh_objects if o.type == 'MESH'])} mesh object(s)")
    return arm_obj
