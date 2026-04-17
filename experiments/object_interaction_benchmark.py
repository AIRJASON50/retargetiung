"""
Object-interaction benchmark for hand retargeting.

Evaluates retargeting quality in hand-object interaction scenarios.
Extends the base RetargetBenchmark with contact and penetration metrics
from the CG motion retargeting literature.

Metrics:
  Contact recall / precision / accuracy  (Jang 2024, SIGGRAPH Asia)
  Tip-object distance error              (adapted from GeoRT L_pinch, IROS 2025)
  Penetration rate and depth             (MeshRet NeurIPS 2024, ReConForM EG 2025)
  Grasp-phase breakdown

Usage:
    PYTHONPATH=src python experiments/object_interaction_benchmark.py
    PYTHONPATH=src python experiments/object_interaction_benchmark.py \\
        --clips experiments/hocap_pipeline/clip_screening/clean_clips.json \\
        --cache data/cache/hocap/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "experiments"))   # for benchmark module
sys.path.insert(0, "/home/l/ws/doc/WujiRepo/wuji_retargeting_private/public")

from benchmark import RetargetBenchmark  # noqa: E402
from hand_retarget.mediapipe_io import load_hocap_clip  # noqa: E402

URDF_ROOT = Path(
    "/home/l/ws/doc/WujiRepo/wuji_retargeting_private/public"
    "/wuji_retargeting/wuji_hand_description/urdf"
)
HOCAP_DIR = PROJECT_DIR / "data" / "hocap" / "hocap"


class ObjectInteractionBenchmark:
    """
    Evaluate hand retargeting quality for hand-object interaction.

    Composes RetargetBenchmark for base hand-shape metrics, adds object-aware
    metrics from CG literature (Jang 2024, GeoRT, MeshRet, ReConForM).

    Coordinate convention:
      - All positions in "hand frame": wrist at origin, MANO-canonical orientation.
      - Object pose (obj_t_hand, obj_q_hand) must be pre-transformed to this frame.
      - Use evaluate_hocap_clip() for automatic frame conversion from cached retarget.
    """

    def __init__(
        self,
        urdf_path: str,
        hand_side: str = "left",
        contact_threshold_mm: float = 15.0,
        robot_contact_threshold_mm: float = 20.0,
        grasp_min_fingers: int = 2,
        scene_xml: str | None = None,
    ):
        self._base = RetargetBenchmark(urdf_path, hand_side)
        self.contact_threshold_mm = contact_threshold_mm
        self.robot_contact_threshold_mm = robot_contact_threshold_mm
        self.grasp_min_fingers = grasp_min_fingers
        self._mesh_cache: dict[str, trimesh.Trimesh] = {}
        self._side = hand_side
        self._mj_hand = None
        if scene_xml is not None:
            from hand_retarget.mujoco_hand import MuJoCoFloatingHandModel
            self._mj_hand = MuJoCoFloatingHandModel(scene_xml, hand_side)

    # ---- Primary API ----

    def evaluate(
        self,
        source_landmarks: np.ndarray,   # (T, 21, 3) hand frame (wrist at origin)
        qpos_seq: np.ndarray,           # (T, nq) joint angles only (no floating-base DOFs)
        obj_t_hand: np.ndarray,         # (T, 3) object center in hand frame
        obj_q_hand: np.ndarray,         # (T, 4) object rotation in hand frame (xyzw)
        mesh_path: str,
        fps: float = 30.0,
        robot_tips: np.ndarray | None = None,   # (T, 5, 3) precomputed FK tips; overrides Pinocchio
    ) -> dict:
        """
        Evaluate retargeting on a hand-object interaction sequence.

        Args:
            source_landmarks: (T, 21, 3) preprocessed MediaPipe landmarks in hand frame.
            qpos_seq: (T, nq) joint angles. Must match URDF DOF (no free-joint prefix).
            obj_t_hand: (T, 3) object center in hand frame.
            obj_q_hand: (T, 4) object rotation quaternion (xyzw) in hand frame.
            mesh_path: path to object mesh (STL/OBJ) in object-local frame.
            fps: frame rate for smoothness metrics.
            robot_tips: (T, 5, 3) precomputed FK tips in hand frame.  If provided,
                skips Pinocchio FK (use with MuJoCo floating-base tips for correct frame).

        Returns:
            dict with keys "base" (RetargetBenchmark output) and
            "object_interaction" (contact / penetration / grasp-phase metrics).
        """
        T = len(qpos_seq)
        assert len(source_landmarks) == T

        base = self._base.evaluate(source_landmarks, qpos_seq, fps)

        mesh = self._load_mesh(mesh_path)

        # Pre-compute FK tips: (T, 5, 3) in hand frame
        if robot_tips is None:
            robot_tips = np.zeros((T, 5, 3))
            for t in range(T):
                robot_tips[t], _, _ = self._base._fk(qpos_seq[t])

        src_tips = source_landmarks[:, RetargetBenchmark.TIP_INDICES, :]   # (T, 5, 3)

        # Rotation matrices for all frames (vectorized below)
        R_objs = np.array([Rotation.from_quat(q).as_matrix() for q in obj_q_hand])  # (T, 3, 3)

        # Signed distances in mm: positive = outside, negative = penetrating
        robot_dists = self._compute_dists(robot_tips, mesh, obj_t_hand, R_objs)   # (T, 5)
        source_dists = self._compute_dists(src_tips,   mesh, obj_t_hand, R_objs)  # (T, 5)

        # Contact labels per finger per frame
        src_contact = source_dists < self.contact_threshold_mm         # (T, 5)
        rob_contact = robot_dists  < self.robot_contact_threshold_mm   # (T, 5)

        src_any = src_contact.any(axis=1)   # (T,) any source finger near object
        rob_any = rob_contact.any(axis=1)   # (T,) any robot finger near object

        # Jang 2024 contact classification
        TP = int((src_any & rob_any).sum())
        FP = int((~src_any & rob_any).sum())
        FN = int((src_any & ~rob_any).sum())
        TN = int((~src_any & ~rob_any).sum())

        recall    = TP / (TP + FN) if (TP + FN) > 0 else 1.0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 1.0
        accuracy  = (TP + TN) / T

        # GeoRT-style tip-object distance error
        dist_error = np.abs(robot_dists - source_dists)   # (T, 5) mm

        # Penetration metrics (MeshRet / ReConForM)
        pen_per_finger = robot_dists < 0          # (T, 5)
        has_pen = pen_per_finger.any(axis=1)      # (T,)
        pen_rate = float(has_pen.mean())
        if pen_per_finger.any():
            pen_depths = -robot_dists[pen_per_finger]   # positive values
            pen_mean_mm = float(pen_depths.mean())
            pen_max_mm  = float(pen_depths.max())
        else:
            pen_mean_mm = 0.0
            pen_max_mm  = 0.0

        # Grasp-phase: >= grasp_min_fingers source fingers near object
        grasp_mask = src_contact.sum(axis=1) >= self.grasp_min_fingers  # (T,)
        n_grasp = int(grasp_mask.sum())

        if n_grasp > 0:
            src_any_g = src_any[grasp_mask]
            rob_any_g = rob_any[grasp_mask]
            TP_g = int((src_any_g & rob_any_g).sum())
            FN_g = int((src_any_g & ~rob_any_g).sum())
            recall_g  = TP_g / (TP_g + FN_g) if (TP_g + FN_g) > 0 else 1.0

            grasp_idx   = np.where(grasp_mask)[0]
            tip_err_g   = float(
                np.linalg.norm(robot_tips[grasp_idx] - src_tips[grasp_idx], axis=-1).mean() * 1000
            )
            pen_rate_g  = float(has_pen[grasp_mask].mean())
        else:
            recall_g   = 0.0
            tip_err_g  = 0.0
            pen_rate_g = 0.0

        # Overall tip position error (FK tips vs source tips, in mm)
        tip_pos_err_all = float(np.linalg.norm(robot_tips - src_tips, axis=-1).mean() * 1000)

        oi = {
            "contact_recall":    float(recall),
            "contact_precision": float(precision),
            "contact_accuracy":  float(accuracy),
            "tip_obj_dist_error_mm": {
                "mean": float(dist_error.mean()),
                "per_finger": {
                    n: float(dist_error[:, i].mean())
                    for i, n in enumerate(RetargetBenchmark.FINGER_NAMES)
                },
            },
            "penetration": {
                "rate":          pen_rate,
                "mean_depth_mm": pen_mean_mm,
                "max_depth_mm":  pen_max_mm,
            },
            "grasp_phase": {
                "n_frames":         n_grasp,
                "fraction":         float(n_grasp / T),
                "contact_recall":   float(recall_g),
                "tip_error_mm":     float(tip_err_g),
                "penetration_rate": float(pen_rate_g),
            },
            "tip_pos_error_mm":           tip_pos_err_all,
            "contact_threshold_mm":       self.contact_threshold_mm,
            "robot_contact_threshold_mm": self.robot_contact_threshold_mm,
            "n_frames": T,
        }

        return {"base": base, "object_interaction": oi}

    # ---- Convenience wrapper for HO-Cap cache ----

    def evaluate_hocap_clip(
        self,
        clip: dict,              # from load_hocap_clip()
        qpos_seq: np.ndarray,    # (T, nq_full) from batch cache (may include floating-base)
        R_inv_seq: np.ndarray,   # (T, 3, 3)  R_align.T per frame
        wrist_seq: np.ndarray,   # (T, 3)     world-frame wrist positions
        fps: float = 30.0,
    ) -> dict:
        """
        Evaluate a HO-Cap clip using cached batch-retarget output.

        Handles world→hand frame transformation for both source landmarks and
        object poses.  Strips floating-base DOFs from qpos if present.

        Frame transformation recap (matches retargeter._align_frame):
          R_inv = R_align.T  (stored in cache)
          p_hand_col = R_inv @ (p_world_col - wrist_col)
          P_hand_row = (P_world_row - wrist_row) @ R_inv.T   [row-vector convention]
        """
        T = len(qpos_seq)
        lm_world = clip["landmarks"]   # (T, 21, 3) raw world frame
        assert len(lm_world) == T, f"landmarks/qpos length mismatch: {len(lm_world)} vs {T}"

        # Strip floating-base DOFs: cache may have [pos(3), quat(4), joints(nq)]
        nq = self._base.nq
        if qpos_seq.shape[1] > nq:
            qpos_joints = qpos_seq[:, qpos_seq.shape[1] - nq:]
        else:
            qpos_joints = qpos_seq

        # Source landmarks → hand frame (row-vector convention)
        # R_inv.T == R_align; (T, 21, 3) @ (T, 3, 3) via einsum
        lm_c = lm_world - wrist_seq[:, None, :]                           # (T, 21, 3)
        source_lm_hand = np.einsum("tki,tij->tkj", lm_c,
                                   np.transpose(R_inv_seq, (0, 2, 1)))    # (T, 21, 3)

        # Object → hand frame
        obj_t_hand, obj_q_hand = self._transform_object_to_hand_frame(
            clip["object_t"], clip["object_q"], R_inv_seq, wrist_seq
        )

        # Use MuJoCo FK when available: gives tips in MANO-aligned hand frame
        # (Pinocchio FK has an orientation mismatch for floating-base optimized qpos)
        mj_tips = self._fk_tips_mujoco(qpos_seq) if self._mj_hand is not None else None

        return self.evaluate(source_lm_hand, qpos_joints, obj_t_hand, obj_q_hand,
                             clip["mesh_path"], fps, robot_tips=mj_tips)

    # ---- Private helpers ----

    def _load_mesh(self, mesh_path: str) -> trimesh.Trimesh:
        if mesh_path not in self._mesh_cache:
            loaded = trimesh.load(str(mesh_path), force="mesh")
            if isinstance(loaded, trimesh.Scene):
                loaded = trimesh.util.concatenate(list(loaded.geometry.values()))
            self._mesh_cache[mesh_path] = loaded
        return self._mesh_cache[mesh_path]

    def _compute_dists(
        self,
        tips: np.ndarray,        # (T, 5, 3) in hand frame
        mesh: trimesh.Trimesh,
        obj_t_seq: np.ndarray,   # (T, 3)
        R_obj_seq: np.ndarray,   # (T, 3, 3) rotation matrices (obj→hand, column convention)
    ) -> np.ndarray:             # (T, 5) signed distances in mm
        """
        Signed tip-to-object surface distances per finger per frame.

        Transforms tips to object local frame, then queries trimesh proximity.
        Sign: positive = outside object surface, negative = penetrating.
        """
        # Vectorized transform to object local frame
        # Row convention: p_obj = (p_hand - obj_center) @ R_obj_hand
        tips_c = tips - obj_t_seq[:, None, :]               # (T, 5, 3)
        tips_local = np.einsum("tfi,tij->tfj", tips_c, R_obj_seq)  # (T, 5, 3)
        tips_flat = tips_local.reshape(-1, 3)               # (T*5, 3)

        try:
            # trimesh convention: positive = inside mesh, negative = outside mesh.
            # We flip to: positive = outside (no contact), negative = inside (penetrating).
            d = -trimesh.proximity.signed_distance(mesh, tips_flat)   # (T*5,) meters
        except Exception:
            # Fallback for non-watertight meshes (already in our convention)
            _, d, _ = trimesh.proximity.closest_point(mesh, tips_flat)
            inside = mesh.contains(tips_flat)
            d = np.where(inside, -d, d)

        return d.reshape(len(obj_t_seq), 5) * 1000   # mm

    def _fk_tips_mujoco(self, qpos_full: np.ndarray) -> np.ndarray:
        """Compute fingertip positions via MuJoCo floating-base FK.

        qpos_full: (T, 26) full floating-base qpos (3 slide + 3 hinge + 20 joints).
        Returns (T, 5, 3) tip positions in MANO-aligned hand frame.
        Thumb=f1, Index=f2, Middle=f3, Ring=f4, Pinky=f5.
        """
        T = len(qpos_full)
        tips = np.zeros((T, 5, 3))
        tip_names = [f"{self._side}_finger{f}_tip_link" for f in range(1, 6)]
        for t in range(T):
            self._mj_hand.forward(qpos_full[t])
            for f, name in enumerate(tip_names):
                tips[t, f] = self._mj_hand.get_body_pos(name)
        return tips

    def _transform_object_to_hand_frame(
        self,
        obj_t_world: np.ndarray,   # (T, 3)
        obj_q_world: np.ndarray,   # (T, 4) xyzw
        R_inv_seq: np.ndarray,     # (T, 3, 3) = R_align.T
        wrist_seq: np.ndarray,     # (T, 3) world-frame wrist positions
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Transform object pose from world frame to hand frame.

        Position (column vectors): obj_t_hand = R_inv @ (obj_t_world - wrist)
        Rotation (R_inv = R_align.T):  R_obj_hand = R_inv @ R_obj_world
        """
        T = len(obj_t_world)
        obj_t_hand = np.zeros((T, 3))
        obj_q_hand = np.zeros((T, 4))

        for t in range(T):
            obj_t_hand[t] = R_inv_seq[t] @ (obj_t_world[t] - wrist_seq[t])
            R_w = Rotation.from_quat(obj_q_world[t]).as_matrix()
            R_h = R_inv_seq[t] @ R_w
            obj_q_hand[t] = Rotation.from_matrix(R_h).as_quat()   # xyzw

        return obj_t_hand, obj_q_hand


# ---- Printing helpers ----

def print_clip_results(results: dict, label: str = "") -> None:
    """Pretty-print combined benchmark results for a single clip."""
    if label:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")

    base = results["base"]
    oi   = results["object_interaction"]

    print(f"Tip Pos Error:    {oi['tip_pos_error_mm']:.2f} mm")
    print(f"JL Violation:     {base['joint_limit_violation_rate']*100:.1f}%")

    print(f"\nContact  (src < {oi['contact_threshold_mm']:.0f}mm / "
          f"robot < {oi['robot_contact_threshold_mm']:.0f}mm):")
    print(f"  Recall      {oi['contact_recall']:.3f}")
    print(f"  Precision   {oi['contact_precision']:.3f}")
    print(f"  Accuracy    {oi['contact_accuracy']:.3f}")

    te = oi["tip_obj_dist_error_mm"]
    print(f"\nTip-Obj Dist Error: {te['mean']:.2f} mm")
    for fname, fval in te["per_finger"].items():
        print(f"  {fname:<10} {fval:>6.2f} mm")

    pen = oi["penetration"]
    print(f"\nPenetration: rate={pen['rate']*100:.1f}%  "
          f"mean={pen['mean_depth_mm']:.2f}mm  max={pen['max_depth_mm']:.2f}mm")

    gp = oi["grasp_phase"]
    print(f"\nGrasp Phase ({gp['n_frames']}f / {gp['fraction']*100:.0f}%):")
    print(f"  Recall   {gp['contact_recall']:.3f}")
    print(f"  Tip Err  {gp['tip_error_mm']:.2f} mm")
    print(f"  Pen Rate {gp['penetration_rate']*100:.1f}%")


def print_summary_table(all_results: list[dict]) -> None:
    """Print aggregate summary across all clips."""
    if not all_results:
        print("No results to summarize.")
        return

    ois  = [r["object_interaction"] for r in all_results]
    base = [r["base"] for r in all_results]

    def stats(vals: list[float]) -> tuple[float, float, float, float]:
        a = np.array(vals, dtype=float)
        return float(a.mean()), float(a.std()), float(a.min()), float(a.max())

    fmt_hdr = "{:<38} {:>8} {:>7} {:>7} {:>7}"
    fmt_row = "{:<38} {:>8.3f} {:>7.3f} {:>7.3f} {:>7.3f}"
    fmt_mm  = "{:<38} {:>8.1f} {:>7.1f} {:>7.1f} {:>7.1f}"

    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  OBJECT INTERACTION BENCHMARK  ({len(all_results)} clips)")
    print(sep)
    print(fmt_hdr.format("Metric", "Mean", "Std", "Min", "Max"))
    print("-" * 72)

    rows_pct = [
        ([o["contact_recall"]    for o in ois], "Contact Recall"),
        ([o["contact_precision"] for o in ois], "Contact Precision"),
        ([o["contact_accuracy"]  for o in ois], "Contact Accuracy"),
        ([o["penetration"]["rate"] for o in ois], "Penetration Rate"),
        ([o["grasp_phase"]["contact_recall"] for o in ois], "[Grasp] Contact Recall"),
        ([o["grasp_phase"]["penetration_rate"] for o in ois], "[Grasp] Pen Rate"),
    ]
    rows_mm = [
        ([o["tip_pos_error_mm"]              for o in ois], "Tip Pos Error (mm)"),
        ([o["tip_obj_dist_error_mm"]["mean"] for o in ois], "Tip-Obj Dist Error (mm)"),
        ([o["penetration"]["mean_depth_mm"]  for o in ois], "Pen Depth mean (mm)"),
        ([o["penetration"]["max_depth_mm"]   for o in ois], "Pen Depth max (mm)"),
        ([o["grasp_phase"]["tip_error_mm"]   for o in ois], "[Grasp] Tip Error (mm)"),
    ]

    for vals, label in rows_pct:
        print(fmt_row.format(label, *stats(vals)))
    print()
    for vals, label in rows_mm:
        print(fmt_mm.format(label, *stats(vals)))

    print(sep)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object interaction benchmark on HO-Cap clips")
    parser.add_argument(
        "--clips", type=str,
        default=str(PROJECT_DIR / "experiments" / "hocap_pipeline" / "clip_screening"
                    / "clean_clips.json"),
        help="Path to clean_clips.json",
    )
    parser.add_argument(
        "--cache", type=str,
        default=str(PROJECT_DIR / "data" / "cache" / "hocap"),
        help="Directory with cached qpos/R_inv npz files",
    )
    parser.add_argument(
        "--hocap", type=str,
        default=str(HOCAP_DIR),
        help="HO-Cap dataset root (must contain motions/ and assets/)",
    )
    parser.add_argument(
        "--contact-thr", type=float, default=15.0, help="Source contact threshold (mm)"
    )
    parser.add_argument(
        "--robot-contact-thr", type=float, default=20.0, help="Robot contact threshold (mm)"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print per-clip detailed results"
    )
    args = parser.parse_args()

    with open(args.clips) as f:
        clean_clips = json.load(f)

    hocap_dir = Path(args.hocap)
    cache_dir = Path(args.cache)

    if not hocap_dir.exists():
        print(f"ERROR: HO-Cap directory not found: {hocap_dir}")
        sys.exit(1)

    scene_xml_map = {
        "right": str(PROJECT_DIR / "assets" / "scenes" / "single_hand_obj.xml"),
        "left":  str(PROJECT_DIR / "assets" / "scenes" / "single_hand_obj_left.xml"),
    }

    # Build one benchmark per hand side (avoid re-initializing Pinocchio models)
    benches: dict[str, ObjectInteractionBenchmark] = {}
    for side in ("left", "right"):
        urdf = str(URDF_ROOT / f"{side}.urdf")
        benches[side] = ObjectInteractionBenchmark(
            urdf, side,
            contact_threshold_mm=args.contact_thr,
            robot_contact_threshold_mm=args.robot_contact_thr,
            scene_xml=scene_xml_map[side],
        )

    all_results: list[dict] = []
    n_skip = 0

    for entry in clean_clips:
        clip_id   = entry["clip_id"]
        hand_side = entry["hand_side"]

        cache_path = cache_dir / f"{clip_id}.npz"
        npz_path   = hocap_dir / "motions" / f"{clip_id}.npz"
        meta_path  = hocap_dir / "motions" / f"{clip_id}.meta.json"

        if not cache_path.exists():
            print(f"[skip] {clip_id} ({hand_side}): cache missing — run batch_retarget_hocap.py")
            n_skip += 1
            continue

        try:
            clip = load_hocap_clip(
                str(npz_path), str(meta_path),
                str(hocap_dir / "assets"),
                hand_side=hand_side,
                sample_count=50,
            )

            cache      = np.load(str(cache_path), allow_pickle=True)
            qpos_seq   = cache[f"qpos_{hand_side}"]
            R_inv_seq  = cache[f"R_inv_{hand_side}"]
            wrist_seq  = cache[f"wrist_{hand_side}"]

            results = benches[hand_side].evaluate_hocap_clip(
                clip, qpos_seq, R_inv_seq, wrist_seq, fps=float(clip["fps"])
            )
            all_results.append(results)

            oi = results["object_interaction"]
            print(
                f"{clip_id[:50]:<50} ({hand_side})  "
                f"recall={oi['contact_recall']:.3f}  "
                f"pen={oi['penetration']['rate']:.3f}  "
                f"grasp={oi['grasp_phase']['n_frames']}f"
            )

            if args.verbose:
                print_clip_results(results, f"{clip_id} ({hand_side})")

        except Exception as exc:
            import traceback
            print(f"[ERROR] {clip_id} ({hand_side}): {exc}")
            traceback.print_exc()
            n_skip += 1

    print_summary_table(all_results)
    print(f"\nProcessed: {len(all_results)}, Skipped/Failed: {n_skip}")
