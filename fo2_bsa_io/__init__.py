# FlatOut Blender Tools — https://github.com/RavenDS/flatout-blender-tools
# __init__.py — Blender add-on entry point: import/export of FlatOut 2 .bsa driver animations on a driver armature
"""
FlatOut 2 driver-animation (.bsa) import/export for Blender.

Imports a .bsa interior-driver animation onto a driver armature previously
created by the FO2 BGM import plugin (typically "fo2_driver_armature").

------------------------------------------------------------------------------
.bsa BINARY FORMAT (little-endian)
------------------------------------------------------------------------------
Global header (20 bytes):
    u32  version       (0x20000)
    u32  bone_count    (11 for the driver skeleton)
    u32  unknown0      (observed 0)
    u32  duration      (observed 16000 — max keyframe time)
    u32  unknown1      (observed 4800 — meaning unconfirmed; unused here)

Then `bone_count` bone blocks, in the same order as bones.ini
Bones[] list (0 = lower_torso, 1 = upper_leg_l, ... 10 = lower_arm_r). Each:
    u32   count          number of keyframes for this bone
    i32   parent_index   parent bone in THIS file's hierarchy (-1 = root).
                         The .bsa hierarchy is rooted at lower_torso and is
                         distinct from the ragdoll constraint hierarchy.
    f32   model_to_bone[16]   ROW-MAJOR 4x4 inverse-bind (ModelToBone). This
                              is a full rigid transform: identity rotation for
                              the legs/torso/head, but a NON-identity rotation
                              for the arms (their rest frames are rotated ~90°,
                              per bones.ini Orientation). The rest model pose is
                              ModelToBone^-1.
    keyframe[count]      32 bytes each:
        u32   time        animation time (0, 160, 320, ... step 160)
        f32   pos[3]      LOCAL translation relative to the parent bone
        f32   quat[4]     LOCAL rotation relative to the parent, (x, y, z, w)

Keyframes are local FK. A bone's animated model transform is:
    M_anim[bone] = M_anim[parent] @ (Translation(pos) @ quat.to_matrix())
chained from the root via parent_index.

The game skins a vertex as  v_anim = M_anim @ ModelToBone @ v_rest, i.e. the
per-bone deformation is  D = M_anim @ ModelToBone.  To reproduce this on the
imported Blender armature (whose rest matrices come from the BGM importer, not
from ModelToBone), each pose bone's armature-space matrix is:
    pose = convert(M_anim @ ModelToBone) @ bone.matrix_local
where convert() is the FO2->Blender axis swap (a Y/Z reflection) plus uniform
scale. This is then turned into a parent-relative matrix_basis and keyframed.

The root bone (lower_torso) stores a single keyframe; all others store the
full timeline. Shorter tracks hold their last value.
------------------------------------------------------------------------------
"""

import os
import struct

import bpy
from bpy.props import StringProperty, FloatProperty, BoolProperty, IntProperty
from bpy_extras.io_utils import ImportHelper, ExportHelper
from mathutils import Vector, Matrix, Quaternion

bl_info = {
    "name": "FlatOut 2 Driver Animation (.bsa)",
    "author": "ravenDS",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "File > Import/Export > FO2 Driver Animation (.bsa)",
    "description": "Import and export FO2 .bsa driver animations on a driver armature",
    "category": "Import-Export",
    "doc_url":     "https://github.com/RavenDS",
    "tracker_url": "https://github.com/RavenDS/flatout-blender-tools/issues",
}

# ============================================================================
#  EXPORT — editable header constants
#  The two .bsa header fields whose meaning is unconfirmed are written verbatim
#  from these values on export. Stock FO2 driver .bsa files use 0 and 4800.
#  Change them here if a particular target build needs different values.
# ============================================================================
EXPORT_UNKNOWN0 = 0
EXPORT_UNKNOWN1 = 4800

BSA_VERSION = 0x20000
KEYFRAME_SIZE = 32
HEADER_SIZE = 20
BLOCK_PREFIX = 8            # count (u32) + parent (i32)
MODEL_TO_BONE_SIZE = 64     # 4x4 float matrix
DEFAULT_ARMATURE = "fo2_driver_armature"

# ============================================================================
#  EXPORT — fixed driver-skeleton tables (nothing is stored on the armature)
#  The driver skeleton is constant, so the exporter rebuilds the .bsa structure
#  from these tables plus the armature's bone positions:
#    * BSA_BONE_ORDER  — .bsa slot order = bones.ini Bones[] order.
#    * BSA_PARENTS     — .bsa parent index per slot (root = lower_torso = -1).
#    * BSA_MTB_ROT     — column-major rotation part of ModelToBone (= inverse of
#                        each bone's FO2 rest orientation). This is universal:
#                        byte-identical for male and female, identity for the
#                        legs/torso/head and a 90deg frame-swap for the arms
#                        (mirrored L vs R). The only per-driver part of
#                        ModelToBone is its translation, which is rebuilt from
#                        the bone's rest head position as  -(R_inv @ rest_pos).
# ============================================================================
BSA_BONE_ORDER = (
    "lower_torso", "upper_leg_l", "lower_leg_l", "upper_leg_r", "lower_leg_r",
    "upper_torso", "head", "upper_arm_l", "lower_arm_l", "upper_arm_r", "lower_arm_r",
)
BSA_PARENTS = (-1, 0, 1, 0, 3, 0, 5, 5, 7, 5, 9)

_MTB_ROT_IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
_MTB_ROT_ARM_L    = ((0.0, -1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, -1.0))
_MTB_ROT_ARM_R    = ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
BSA_MTB_ROT = (
    _MTB_ROT_IDENTITY,  # lower_torso
    _MTB_ROT_IDENTITY,  # upper_leg_l
    _MTB_ROT_IDENTITY,  # lower_leg_l
    _MTB_ROT_IDENTITY,  # upper_leg_r
    _MTB_ROT_IDENTITY,  # lower_leg_r
    _MTB_ROT_IDENTITY,  # upper_torso
    _MTB_ROT_IDENTITY,  # head
    _MTB_ROT_ARM_L,     # upper_arm_l
    _MTB_ROT_ARM_L,     # lower_arm_l
    _MTB_ROT_ARM_R,     # upper_arm_r
    _MTB_ROT_ARM_R,     # lower_arm_r
)

# FO2 -> Blender axis swap (x, y, z) -> (x, z, y); a Y/Z reflection (det = -1).
_AXIS_SWAP = Matrix((
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
))


def _fo2_to_bl(fo2_pos, scale=1.0):
    """FO2 (x, y, z) -> Blender (x, z, y); same axis convention as the importer."""
    x, y, z = fo2_pos
    return Vector((x * scale, z * scale, y * scale))


def _bl_to_fo2(bl_pos, scale=1.0):
    """Blender (x, y, z) -> FO2 (x, z, y) / scale; inverse of `_fo2_to_bl`."""
    return Vector((bl_pos.x / scale, bl_pos.z / scale, bl_pos.y / scale))


def _convert_matrix(m_fo2, scale=1.0):
    """Convert an FO2 model-space 4x4 (column-major mathutils) to Blender.

    Conjugates by the Y/Z axis swap and scales the translation by `scale`,
    matching how the BGM importer placed geometry and bones.
    """
    mb = _AXIS_SWAP @ m_fo2 @ _AXIS_SWAP
    mb.translation = mb.translation * scale
    return mb


def _convert_matrix_inv(mb, scale=1.0):
    """Inverse of `_convert_matrix`: Blender model-space 4x4 -> FO2 model-space.

    Undoes the translation scaling, then conjugates by the (self-inverse) Y/Z
    axis swap. Used by the exporter to recover FO2-space transforms from the
    posed Blender bones.
    """
    pre = mb.copy()
    pre.translation = mb.translation / scale
    return _AXIS_SWAP @ pre @ _AXIS_SWAP


#  Parser
class BsaBone:
    __slots__ = ("parent", "model_to_bone", "model_to_bone_raw",
                 "rest_pos_fo2", "times", "locs", "quats")


class BsaAnim:
    __slots__ = ("version", "duration", "bones")


def parse_bsa(filepath):
    """Parse a .bsa file into a BsaAnim. Raises ValueError on malformed data."""
    with open(filepath, "rb") as fh:
        data = fh.read()

    if len(data) < HEADER_SIZE:
        raise ValueError("File too small to be a .bsa")

    version, bone_count, _u0, duration, _u1 = struct.unpack_from("<5I", data, 0)
    if version != BSA_VERSION:
        print("[bsa] warning: unexpected version 0x%x (expected 0x%x)"
              % (version, BSA_VERSION))
    if not (1 <= bone_count <= 256):
        raise ValueError("Implausible bone_count %d" % bone_count)

    anim = BsaAnim()
    anim.version = version
    anim.duration = duration
    anim.bones = []

    off = HEADER_SIZE
    for bi in range(bone_count):
        if off + BLOCK_PREFIX + MODEL_TO_BONE_SIZE > len(data):
            raise ValueError("Truncated bone block %d" % bi)
        count, parent = struct.unpack_from("<Ii", data, off)
        off += BLOCK_PREFIX
        m = struct.unpack_from("<16f", data, off)
        off += MODEL_TO_BONE_SIZE

        # Stored ROW-MAJOR (translation in the last row). mathutils is
        # column-major, so transpose to get the native matrix.
        mtb = Matrix((m[0:4], m[4:8], m[8:12], m[12:16])).transposed()
        rest = mtb.inverted_safe().translation  # rest model position (incl. arm rotation)

        if count < 1 or off + count * KEYFRAME_SIZE > len(data):
            raise ValueError("Bad keyframe count %d in bone block %d" % (count, bi))

        times, locs, quats = [], [], []
        for _ in range(count):
            t, px, py, pz, qx, qy, qz, qw = struct.unpack_from("<I7f", data, off)
            off += KEYFRAME_SIZE
            times.append(t)
            locs.append((px, py, pz))
            quats.append((qx, qy, qz, qw))

        b = BsaBone()
        b.parent = parent
        b.model_to_bone = mtb
        b.model_to_bone_raw = list(m)   # row-major 16 floats, exactly as on disk
        b.rest_pos_fo2 = (rest.x, rest.y, rest.z)
        b.times = times
        b.locs = locs
        b.quats = quats
        anim.bones.append(b)

    return anim



#  FK reconstruction (FO2 model space, column-major mathutils)
def _local_matrix(bone, k):
    px, py, pz = bone.locs[k]
    qx, qy, qz, qw = bone.quats[k]
    rot = Quaternion((qw, qx, qy, qz)).to_matrix().to_4x4()  # mathutils takes (w,x,y,z)
    return Matrix.Translation(Vector((px, py, pz))) @ rot


def reconstruct_model_matrices(bones, frame_index):
    """Animated model-space 4x4 for every bone at `frame_index` (hold-last)."""
    n = len(bones)
    cache = [None] * n

    def solve(i):
        if cache[i] is not None:
            return cache[i]
        b = bones[i]
        k = frame_index if frame_index < len(b.times) else len(b.times) - 1
        local = _local_matrix(b, k)
        cache[i] = local if b.parent < 0 else (solve(b.parent) @ local)
        return cache[i]

    return [solve(i) for i in range(n)]


#  Bone name mapping (.bsa stores no names — match by rest head position)
def map_bsa_to_blender(anim, armature, scale, tol=1e-3):
    """Return ({bsa_index: blender_bone_name}, [unmatched...]) by rest position."""
    mapping = {}
    unmatched = []
    bones = armature.data.bones
    for i, b in enumerate(anim.bones):
        target = _fo2_to_bl(b.rest_pos_fo2, scale)
        best_name, best_d = None, float("inf")
        for bone in bones:
            d = (bone.head_local - target).length
            if d < best_d:
                best_d, best_name = d, bone.name
        mapping[i] = best_name
        if best_d > tol:
            unmatched.append((i, best_name, best_d))
    return mapping, unmatched


#  Operator
class IMPORT_OT_fo2_bsa(bpy.types.Operator, ImportHelper):
    bl_idname = "import_anim.fo2_bsa"
    bl_label = "Import FO2 Driver Animation"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".bsa"
    filter_glob: StringProperty(default="*.bsa", options={"HIDDEN"})

    global_scale: FloatProperty(
        name="Scale",
        description="Must match the scale used when the driver model was imported",
        default=1.0, min=1e-4, max=1e4,
    )
    armature_name: StringProperty(
        name="Armature",
        description="Target driver armature (used if 'Use Active Armature' is off)",
        default=DEFAULT_ARMATURE,
    )
    use_active: BoolProperty(
        name="Use Active Armature",
        description="Apply to the active object if it is an armature",
        default=True,
    )
    match_tolerance: FloatProperty(
        name="Match Tolerance",
        description="Max distance (Blender units) when matching .bsa bones to armature bones",
        default=1e-3, min=1e-6, max=1.0,
    )
    disconnect_spine: BoolProperty(
        name="Disconnect Spine",
        description="Clear use_connect on the spine bones before baking.",
        default=True,
    )

    def _resolve_armature(self, context):
        if self.use_active:
            obj = context.active_object
            if obj is not None and obj.type == "ARMATURE":
                return obj
        obj = bpy.data.objects.get(self.armature_name)
        if obj is not None and obj.type == "ARMATURE":
            return obj
        arms = [o for o in context.scene.objects if o.type == "ARMATURE"]
        if len(arms) == 1:
            return arms[0]
        return None

    def execute(self, context):
        arm = self._resolve_armature(context)
        if arm is None:
            self.report({"ERROR"},
                        "No target armature. Select the driver armature, or set its "
                        "name (default '%s')." % DEFAULT_ARMATURE)
            return {"CANCELLED"}

        try:
            anim = parse_bsa(self.filepath)
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, "Failed to parse .bsa: %s" % exc)
            return {"CANCELLED"}

        scale = self.global_scale
        mapping, unmatched = map_bsa_to_blender(anim, arm, scale, self.match_tolerance)
        if unmatched:
            preview = ", ".join("#%d->%s (%.3f)" % u for u in unmatched[:4])
            self.report({"WARNING"},
                        "%d .bsa bone(s) did not match cleanly (check Scale). %s"
                        % (len(unmatched), preview))

        # Master timeline = the bone with the most keyframes.
        master = max(range(len(anim.bones)), key=lambda i: len(anim.bones[i].times))
        master_times = anim.bones[master].times
        n_frames = len(master_times)
        diffs = [master_times[i + 1] - master_times[i]
                 for i in range(len(master_times) - 1)
                 if master_times[i + 1] > master_times[i]]
        tick = min(diffs) if diffs else 1
        frame_numbers = [round(t / tick) + 1 for t in master_times]

        # Enter pose mode.
        prev_mode = arm.mode
        prev_active = context.view_layer.objects.active
        context.view_layer.objects.active = arm


        if self.disconnect_spine:
            bpy.ops.object.mode_set(mode="EDIT")
            edit_bones = arm.data.edit_bones
            for bn in ("lower_torso", "upper_torso", "head"):
                eb = edit_bones.get(bn)
                if eb is not None and eb.use_connect:
                    eb.use_connect = False
            bpy.ops.object.mode_set(mode="OBJECT")

        bpy.ops.object.mode_set(mode="POSE")
        pose_bones = arm.pose.bones
        for pb in pose_bones:
            pb.rotation_mode = "QUATERNION"

        action_name = "fo2_driver_bsa"
        existing = bpy.data.actions.get(action_name)
        if existing is not None:
            bpy.data.actions.remove(existing)
        arm.animation_data_create()
        action = bpy.data.actions.new(action_name)
        arm.animation_data.action = action

        rest_local = {pb.name: pb.bone.matrix_local for pb in pose_bones}

        for fi in range(n_frames):
            fr = frame_numbers[fi]
            models = reconstruct_model_matrices(anim.bones, fi)

            # Target armature-space pose matrix for every bone (rest where unmapped).
            m_target = {pb.name: rest_local[pb.name].copy() for pb in pose_bones}
            for i, b in enumerate(anim.bones):
                name = mapping.get(i)
                if name in m_target:
                    deform = models[i] @ b.model_to_bone        # D = M_anim @ ModelToBone
                    m_target[name] = _convert_matrix(deform, scale) @ rest_local[name]

            # Absolute -> parent-relative basis (Blender's own hierarchy), keyframe.
            for pb in pose_bones:
                ml = rest_local[pb.name]
                if pb.parent is not None:
                    par_ml = rest_local[pb.parent.name]
                    par_t = m_target.get(pb.parent.name, par_ml)
                    pb.matrix_basis = ml.inverted() @ par_ml @ par_t.inverted() @ m_target[pb.name]
                else:
                    pb.matrix_basis = ml.inverted() @ m_target[pb.name]
                pb.keyframe_insert("location", frame=fr)
                pb.keyframe_insert("rotation_quaternion", frame=fr)

        # Restore state; leave the rig on the first frame.
        scene = context.scene
        scene.frame_start = frame_numbers[0]
        scene.frame_end = frame_numbers[-1]
        scene.frame_set(frame_numbers[0])
        try:
            bpy.ops.object.mode_set(mode=prev_mode)
        except Exception:  # noqa: BLE001
            bpy.ops.object.mode_set(mode="OBJECT")
        if prev_active is not None:
            context.view_layer.objects.active = prev_active

        # Record the bone order (mapped Blender bone names, in .bsa slot order) and the
        # .bsa parent indices so the exporter can reproduce the structure exactly,
        # including a non-standard skeleton. ModelToBone is rebuilt from bone positions
        # on export, so it is not stored.
        n = len(anim.bones)
        arm["bsa_bone_order"] = ",".join((mapping.get(i) or "") for i in range(n))
        arm["bsa_parents"]    = [int(b.parent) for b in anim.bones]

        self.report({"INFO"},
                    "Imported '%s': %d frames onto %d bones (action '%s')."
                    % (os.path.basename(self.filepath), n_frames,
                       len(anim.bones) - len(unmatched), action_name))
        return {"FINISHED"}


#  Writer
def write_bsa(filepath, version, unknown0, duration, unknown1, blocks):
    """Write a .bsa file.

    blocks: list of (parent_index, model_to_bone_rowmajor[16],
                     [(time, (px, py, pz), (qx, qy, qz, qw)), ...]).
    """
    buf = bytearray()
    buf += struct.pack("<5I", int(version) & 0xFFFFFFFF, len(blocks),
                       int(unknown0) & 0xFFFFFFFF, int(duration) & 0xFFFFFFFF,
                       int(unknown1) & 0xFFFFFFFF)
    for parent, mtb16, keys in blocks:
        buf += struct.pack("<Ii", len(keys), int(parent))
        buf += struct.pack("<16f", *(float(x) for x in mtb16))
        for t, pos, quat in keys:
            buf += struct.pack("<I7f", int(t),
                               float(pos[0]), float(pos[1]), float(pos[2]),
                               float(quat[0]), float(quat[1]),
                               float(quat[2]), float(quat[3]))
    with open(filepath, "wb") as fh:
        fh.write(buf)


#  Export operator
class EXPORT_OT_fo2_bsa(bpy.types.Operator, ExportHelper):
    """Export the active action on a driver armature back to a FO2 .bsa.

    Requires an armature that was set up by importing a .bsa (the import records
    the bone order, .bsa parent hierarchy and inverse-bind matrices needed to
    write a valid file). The animation is read by sampling the action per frame
    and inverting the importer's transform pipeline.
    """
    bl_idname = "export_anim.fo2_bsa"
    bl_label = "Export FO2 Driver Animation"
    bl_options = {"REGISTER"}

    filename_ext = ".bsa"
    filter_glob: StringProperty(default="*.bsa", options={"HIDDEN"})

    armature_name: StringProperty(
        name="Armature",
        description="Source driver armature (used if 'Use Active Armature' is off)",
        default=DEFAULT_ARMATURE,
    )
    use_active: BoolProperty(
        name="Use Active Armature",
        description="Export from the active object if it is an armature",
        default=True,
    )
    global_scale: FloatProperty(
        name="Scale",
        description="Must match the scale used when the .bsa / driver model was imported",
        default=1.0, min=1e-4, max=1e4,
    )
    tick: IntProperty(
        name="Time Step",
        description="Animation time units per frame written to the .bsa "
                    "(stock FO2 driver animations use 160)",
        default=160, min=1,
    )
    version: IntProperty(
        name="Version",
        description="Value written to the .bsa version header field "
                    "(stock FO2 driver animations use 131072)",
        default=131072, min=0,
    )

    def execute(self, context):
        # Reuse the importer's armature resolver (same property names).
        arm = IMPORT_OT_fo2_bsa._resolve_armature(self, context)
        if arm is None:
            self.report({"ERROR"},
                        "No source armature. Select the driver armature, or set its "
                        "name (default '%s')." % DEFAULT_ARMATURE)
            return {"CANCELLED"}

        ad = arm.animation_data
        action = ad.action if ad else None
        if action is None:
            self.report({"ERROR"}, "Armature has no active action to export.")
            return {"CANCELLED"}

        # Bone order and .bsa parents: prefer values recorded at import (so a
        # non-standard skeleton round-trips exactly); otherwise fall back to the fixed
        # driver-skeleton tables. ModelToBone is always rebuilt from bone positions.
        # Version, scale and tick come from the operator. The single-keyframe bone is
        # always the root (parent == -1).
        if "bsa_bone_order" in arm:
            bone_order = list(arm["bsa_bone_order"].split(","))
        else:
            bone_order = list(BSA_BONE_ORDER)
        if "bsa_parents" in arm:
            parents = [int(p) for p in arm["bsa_parents"]]
        else:
            parents = list(BSA_PARENTS)
        n_bones    = len(bone_order)
        version    = self.version
        scale      = self.global_scale
        tick       = self.tick

        pose_bones = arm.pose.bones
        missing = [bn for bn in bone_order if bn not in pose_bones]
        if missing:
            self.report({"ERROR"}, "Armature is missing bone(s): %s" % ", ".join(missing))
            return {"CANCELLED"}

        # ModelToBone per bone. If that property is absent or incomplete,
        # rebuild each matrix from the universal rotation table plus the bone's rest head position:
        #   MTB = R_inv @ T(-rest_pos)   ->   rotation = R_inv,  translation = -(R_inv @ rest_pos)
        # 'r' below is the row-major form written to file; rest_mat is its column-major
        # inverse (the model-space rest transform) used by the keyframe loop.
        mtb_prop = arm.get("fo2_driver_mtb")
        use_stored_mtb = False
        if mtb_prop is not None and hasattr(mtb_prop, "keys"):
            stored_names = set(mtb_prop.keys())
            use_stored_mtb = all(bn in stored_names for bn in bone_order)

        mtb_raw = []
        rest_mat = []
        for i, bn in enumerate(bone_order):
            if use_stored_mtb:
                r = [float(v) for v in mtb_prop[bn]]
            else:
                rest_pos = _bl_to_fo2(pose_bones[bn].bone.head_local, scale)
                r_inv = Matrix(BSA_MTB_ROT[i]).to_3x3()
                mtb_cm = r_inv.to_4x4()
                mtb_cm.translation = -(r_inv @ rest_pos)
                rows = mtb_cm.transposed()
                r = [rows[rr][cc] for rr in range(4) for cc in range(4)]
            mtb_raw.append(r)
            rest_mat.append(
                Matrix((r[0:4], r[4:8], r[8:12], r[12:16])).transposed().inverted_safe())

        rest_local = {bn: pose_bones[bn].bone.matrix_local for bn in bone_order}

        # Frame range from the action; first frame maps to time 0.
        fr0, fr1 = action.frame_range
        first, last = int(round(fr0)), int(round(fr1))
        frames = list(range(first, last + 1))

        scene = context.scene
        prev_frame = scene.frame_current
        prev_active = context.view_layer.objects.active
        prev_mode = arm.mode
        context.view_layer.objects.active = arm
        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except Exception:  # noqa: BLE001
            pass

        per_bone_keys = [[] for _ in range(n_bones)]
        for frame in frames:
            scene.frame_set(frame)
            t = (frame - first) * tick
            # Absolute animated FO2 model transform per bone (invert the importer).
            m_anim = [None] * n_bones
            for i, bn in enumerate(bone_order):
                pb = pose_bones[bn]
                d_bl = pb.matrix @ rest_local[bn].inverted()      # = convert(M_anim @ MTB)
                d_fo2 = _convert_matrix_inv(d_bl, scale)          # = M_anim @ MTB
                m_anim[i] = d_fo2 @ rest_mat[i]                   # = M_anim
            # Local FK relative to the .bsa parent.
            for i in range(n_bones):
                p = parents[i]
                local = m_anim[i] if p < 0 else (m_anim[p].inverted() @ m_anim[i])
                loc = local.to_translation()
                q = local.to_quaternion()
                per_bone_keys[i].append(
                    (t, (loc.x, loc.y, loc.z), (q.x, q.y, q.z, q.w)))

        scene.frame_set(prev_frame)
        if prev_active is not None:
            context.view_layer.objects.active = prev_active
        try:
            bpy.ops.object.mode_set(mode=prev_mode)
        except Exception:  # noqa: BLE001
            pass

        # Assemble blocks. The root bone (parent == -1) keeps a single keyframe,
        # matching the stock .bsa convention; every other bone keeps the timeline.
        blocks = []
        for i in range(n_bones):
            keys = per_bone_keys[i]
            if parents[i] < 0 and keys:
                keys = keys[:1]
            blocks.append((parents[i], mtb_raw[i], keys))
        duration = (last - first) * tick

        try:
            write_bsa(self.filepath, version, EXPORT_UNKNOWN0, duration,
                      EXPORT_UNKNOWN1, blocks)
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, "Failed to write .bsa: %s" % exc)
            return {"CANCELLED"}

        self.report({"INFO"},
                    "Exported '%s': %d bones, %d frames (unknown0=%d, unknown1=%d)."
                    % (os.path.basename(self.filepath), n_bones, len(frames),
                       EXPORT_UNKNOWN0, EXPORT_UNKNOWN1))
        return {"FINISHED"}


def _menu_func_import(self, context):
    self.layout.operator(IMPORT_OT_fo2_bsa.bl_idname,
                         text="FlatOut Driver Animation (.bsa)")


def _menu_func_export(self, context):
    self.layout.operator(EXPORT_OT_fo2_bsa.bl_idname,
                         text="FlatOut Driver Animation (.bsa)")


def register():
    bpy.utils.register_class(IMPORT_OT_fo2_bsa)
    bpy.utils.register_class(EXPORT_OT_fo2_bsa)
    bpy.types.TOPBAR_MT_file_import.append(_menu_func_import)
    bpy.types.TOPBAR_MT_file_export.append(_menu_func_export)


def unregister():
    bpy.types.TOPBAR_MT_file_export.remove(_menu_func_export)
    bpy.types.TOPBAR_MT_file_import.remove(_menu_func_import)
    bpy.utils.unregister_class(EXPORT_OT_fo2_bsa)
    bpy.utils.unregister_class(IMPORT_OT_fo2_bsa)


if __name__ == "__main__":
    register()
