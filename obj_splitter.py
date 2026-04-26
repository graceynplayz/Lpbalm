#!/usr/bin/env python3
"""
OBJ 3D Print Splitter
Cuts a single OBJ mesh into multiple parts along an axis and adds cylindrical
alignment pins/holes at each cut face so the parts fit together for printing.

Usage:
    python obj_splitter.py model.obj -n 2 -a z
    python obj_splitter.py model.obj --max-size 150 -a z --output-dir parts/
"""

import argparse
import os
import sys
import numpy as np

try:
    import trimesh
    import trimesh.boolean as tbool
    import trimesh.intersections as tintersect
except ImportError:
    sys.exit("Missing dependency: pip install trimesh manifold3d numpy")

AXIS_MAP = {"x": 0, "y": 1, "z": 2}


# ---------------------------------------------------------------------------
# Mesh loading
# ---------------------------------------------------------------------------

def load_mesh(path: str) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="mesh", process=True)
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"Could not interpret '{path}' as a single mesh.")
    if not loaded.is_watertight:
        print("  Warning: mesh is not watertight — slicing and booleans may produce gaps.")
    return loaded


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def cut_positions(mesh: trimesh.Trimesh, axis: int, n_parts: int) -> list:
    lo, hi = mesh.bounds[0][axis], mesh.bounds[1][axis]
    return [lo + (hi - lo) * i / n_parts for i in range(1, n_parts)]


def slice_at(mesh: trimesh.Trimesh, axis: int, pos: float):
    """Return (lower, upper) parts split at the given plane position.

    slice_mesh_plane keeps vertices where (v - origin)·normal >= 0, i.e. the
    side the normal *points toward*.  So to get the lower half (axis <= pos)
    we point the normal in the *negative* axis direction.
    """
    normal = np.zeros(3)
    normal[axis] = 1.0
    origin = np.zeros(3)
    origin[axis] = pos

    # lower: keeps axis-coordinate <= pos  (normal points in -axis direction)
    lower = tintersect.slice_mesh_plane(mesh, -normal, origin, cap=True)
    # upper: keeps axis-coordinate >= pos  (normal points in +axis direction)
    upper = tintersect.slice_mesh_plane(mesh,  normal, origin, cap=True)

    if lower is None or not len(lower.faces):
        raise RuntimeError(f"Lower half at {pos:.2f} is empty — check cut position.")
    if upper is None or not len(upper.faces):
        raise RuntimeError(f"Upper half at {pos:.2f} is empty — check cut position.")

    return lower, upper


# ---------------------------------------------------------------------------
# Connector geometry
# ---------------------------------------------------------------------------

def _aligned_cylinder(axis: int, center: np.ndarray, radius: float, height: float,
                       sections: int = 24) -> trimesh.Trimesh:
    """Cylinder aligned with `axis`, centred at `center`."""
    cyl = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    if axis == 0:
        cyl.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    elif axis == 1:
        cyl.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    cyl.apply_translation(center)
    return cyl


def _pin_grid(part: trimesh.Trimesh, axis: int, cut_pos: float,
              spacing: float, eps: float = 1.0) -> list:
    """Return a list of (u, v) positions for connectors on the cut face."""
    other = [i for i in range(3) if i != axis]
    u_ax, v_ax = other

    verts = part.vertices
    near = np.abs(verts[:, axis] - cut_pos) < eps
    if near.sum() < 3:
        return []

    cv = verts[near]
    u_min, u_max = cv[:, u_ax].min(), cv[:, u_ax].max()
    v_min, v_max = cv[:, v_ax].min(), cv[:, v_ax].max()

    u_pts = np.arange(u_min + spacing / 2, u_max, spacing)
    v_pts = np.arange(v_min + spacing / 2, v_max, spacing)
    if len(u_pts) == 0:
        u_pts = [(u_min + u_max) / 2]
    if len(v_pts) == 0:
        v_pts = [(v_min + v_max) / 2]

    return [(float(u), float(v)) for u in u_pts for v in v_pts]


def add_connectors(lower: trimesh.Trimesh, upper: trimesh.Trimesh,
                   axis: int, cut_pos: float,
                   pin_radius: float, pin_height: float,
                   tolerance: float, spacing: float,
                   engine: str) -> tuple:
    """
    lower gets pins protruding above its cut face.
    upper gets matching holes (pin_radius + tolerance) in its cut face.
    Both cylinders are centred at cut_pos + pin_height/2 on the split axis
    so the pin sits exactly in the hole when the parts are joined.
    """
    other = [i for i in range(3) if i != axis]
    u_ax, v_ax = other

    grid = _pin_grid(lower, axis, cut_pos, spacing)
    if not grid:
        print("  Warning: no connector positions found near cut face — skipping connectors.")
        return lower, upper

    print(f"  Placing {len(grid)} connector pin(s)/hole(s)...")

    pins = []
    holes = []
    for u_val, v_val in grid:
        center = np.zeros(3)
        center[axis] = cut_pos + pin_height / 2   # protrudes above cut face
        center[u_ax] = u_val
        center[v_ax] = v_val

        pins.append(_aligned_cylinder(axis, center, pin_radius, pin_height))
        holes.append(_aligned_cylinder(axis, center,
                                        pin_radius + tolerance,
                                        pin_height + 2 * tolerance))

    try:
        lower = tbool.union([lower] + pins, engine=engine)
        upper = tbool.difference([upper] + holes, engine=engine)
        # Clean up after booleans to avoid degenerate geometry on next slice
        lower = trimesh.Trimesh(vertices=lower.vertices, faces=lower.faces, process=True)
        upper = trimesh.Trimesh(vertices=upper.vertices, faces=upper.faces, process=True)
    except Exception as exc:
        print(f"  Warning: boolean ops failed ({exc}).")
        print("  Ensure manifold3d is installed: pip install manifold3d")
        print("  Parts will be exported without connectors for this cut.")

    return lower, upper


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_parts(parts: list, output_dir: str, base: str, fmt: str):
    os.makedirs(output_dir, exist_ok=True)
    for i, mesh in enumerate(parts, 1):
        path = os.path.join(output_dir, f"{base}_part{i:02d}.{fmt}")
        mesh.export(path)
        verts, faces = len(mesh.vertices), len(mesh.faces)
        bb = mesh.bounds[1] - mesh.bounds[0]
        print(f"  [{i}/{len(parts)}] {path}  ({bb[0]:.1f}×{bb[1]:.1f}×{bb[2]:.1f} mm, "
              f"{faces:,} faces)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Split an OBJ into multiple parts with interlocking alignment pins.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", help="Input OBJ file")
    p.add_argument("-n", "--parts", type=int, default=2,
                   help="Number of parts")
    p.add_argument("-a", "--axis", choices=["x", "y", "z"], default="z",
                   help="Axis to split along")
    p.add_argument("--max-size", type=float, default=None,
                   help="Max part size (mm) along split axis; overrides --parts")
    p.add_argument("-o", "--output-dir", default="output_parts",
                   help="Output directory")
    p.add_argument("--format", choices=["obj", "stl"], default="stl",
                   help="Output file format")
    p.add_argument("--pin-radius", type=float, default=2.5,
                   help="Alignment pin radius (mm)")
    p.add_argument("--pin-height", type=float, default=5.0,
                   help="Alignment pin height (mm)")
    p.add_argument("--tolerance", type=float, default=0.2,
                   help="Hole oversize for fit tolerance (mm)")
    p.add_argument("--spacing", type=float, default=25.0,
                   help="Pin grid spacing (mm)")
    p.add_argument("--no-connectors", action="store_true",
                   help="Export plain sliced parts without alignment pins")
    p.add_argument("--engine", default="manifold",
                   help="Boolean engine: manifold (default), blender, or scad")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    axis = AXIS_MAP[args.axis]

    # ---- Load ----
    print(f"Loading: {args.input}")
    mesh = load_mesh(args.input)
    dims = mesh.bounds[1] - mesh.bounds[0]
    print(f"  Dimensions : {dims[0]:.2f} × {dims[1]:.2f} × {dims[2]:.2f} mm")
    print(f"  Triangles  : {len(mesh.faces):,}")
    print(f"  Watertight : {mesh.is_watertight}")

    # ---- Determine cut count ----
    n_parts = args.parts
    if args.max_size is not None:
        span = dims[axis]
        n_parts = max(2, int(np.ceil(span / args.max_size)))
        print(f"  Auto-split : {n_parts} parts "
              f"({span:.1f} mm / {args.max_size} mm max)")

    if n_parts < 2:
        sys.exit("Need at least 2 parts (--parts must be >= 2).")

    cuts = cut_positions(mesh, axis, n_parts)
    print(f"\nSplitting along {args.axis}-axis into {n_parts} part(s).")
    print(f"Cut planes at {args.axis} = {[f'{c:.2f}' for c in cuts]}")

    # ---- Iterative slicing ----
    # Stack works back-to-front: slice the tail piece off the remainder each pass.
    remainder = mesh
    parts = []

    for idx, cut_pos in enumerate(cuts):
        step = idx + 1
        print(f"\nCut {step}/{len(cuts)} at {args.axis} = {cut_pos:.2f} mm ...")
        lower, remainder = slice_at(remainder, axis, cut_pos)

        if not args.no_connectors:
            lower, remainder = add_connectors(
                lower, remainder, axis, cut_pos,
                pin_radius=args.pin_radius,
                pin_height=args.pin_height,
                tolerance=args.tolerance,
                spacing=args.spacing,
                engine=args.engine,
            )

        parts.append(lower)

    parts.append(remainder)  # final (top) piece

    # ---- Export ----
    base = os.path.splitext(os.path.basename(args.input))[0]
    print(f"\nExporting {len(parts)} parts to '{args.output_dir}/' ...")
    export_parts(parts, args.output_dir, base, args.format)

    print(f"\nDone. Print each part then press them together — pins lock into holes.")


if __name__ == "__main__":
    main()
