"""Scan alignment MTF terrain by field and tolerance scenario.

The scan uses the same episode-relative log-MTF objective as the RL environment.
It is intentionally a small, deterministic diagnostic (not a training utility).
Run from ``alignment_rl-dev`` with the project's Python environment::

    python analyze_mtf_terrain.py --grid 9 --rays 32 --output terrain.json

The JSON contains one q surface per field, plus peak/edge/curvature summaries.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from config import LensEnvConfig, LensGroupConfig
from env.lens_env import LensAlignmentEnv


FIELDS = [(0.0, 0.0), (14.0, 0.0), (-14.0, 0.0), (0.0, 14.0), (0.0, -14.0)]


def make_env(tol_scale: float, rays: int) -> LensAlignmentEnv:
    cfg = LensEnvConfig(
        tol_radius_rel=0.001 * tol_scale,
        tol_thickness_mm=0.010 * tol_scale,
        tol_decenter_mm=0.015 * tol_scale,
        tol_tilt_deg=0.03 * tol_scale,
        use_compensator=False,
        mtf_num_rays=rays,
        mtf_field_coords=FIELDS,
        mtf_field_indices=list(range(len(FIELDS))),
        lens_groups=[LensGroupConfig(3, 4, 2, active_dofs=["dx", "dy"])],
    )
    env = LensAlignmentEnv(cfg=cfg)
    env.reset(seed=1234)
    # Reset establishes the sampled tolerance and the zero-pose reference.
    env._mgr.apply_alignment_state(np.zeros(env.n_dof, dtype=np.float64))
    env._mgr.set_episode_reference()
    return env


def scan(env: LensAlignmentEnv, half_range: float, grid: int) -> np.ndarray:
    vals = np.linspace(-half_range, half_range, grid)
    surfaces = np.empty((len(FIELDS), grid, grid), dtype=float)
    ref = env._mgr._episode_ref_mtf_obs
    for iy, dy in enumerate(vals):
        for ix, dx in enumerate(vals):
            env._mgr.apply_alignment_state(np.array([dx, dy], dtype=np.float64))
            raw = env._mgr._compute_mtf_obs()
            rel = env._mgr._relative_log_mtf_obs(raw, ref)
            n = len(FIELDS) * 2 * len(env.cfg.mtf_frequencies)
            surfaces[:, iy, ix] = rel.reshape(len(FIELDS), n // len(FIELDS)).mean(axis=1)
    return vals, surfaces


def summarize(vals: np.ndarray, surfaces: np.ndarray) -> list[dict[str, float]]:
    out = []
    edge = (np.abs(vals[:, None]) == np.max(np.abs(vals))) | (np.abs(vals[None, :]) == np.max(np.abs(vals)))
    for i, q in enumerate(surfaces):
        peak = np.unravel_index(np.argmax(q), q.shape)
        center = q[q.shape[0] // 2, q.shape[1] // 2]
        out.append({
            "field_index": i,
            "peak_q": float(q[peak]),
            "peak_dx_mm": float(vals[peak[1]]),
            "peak_dy_mm": float(vals[peak[0]]),
            "center_q": float(center),
            "edge_median_q": float(np.median(q[edge])),
            "center_to_edge_drop": float(center - np.median(q[edge])),
            "local_curvature_q_per_mm2": float((q[peak[0], min(peak[1] + 1, q.shape[1] - 1)] + q[peak[0], max(peak[1] - 1, 0)] - 2 * q[peak]) / (vals[1] - vals[0]) ** 2),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=7)
    ap.add_argument("--rays", type=int, default=32)
    ap.add_argument("--output", type=Path, default=Path("terrain.json"))
    ap.add_argument("--html", type=Path, default=Path("terrain.html"))
    args = ap.parse_args()
    if args.grid < 3:
        ap.error("--grid must be at least 3")
    result = {"fields_deg": FIELDS, "frequencies_lp_mm": [20.0, 30.0, 50.0], "scenarios": {}}
    for name, scale in (("nominal", 0.0), ("current_tolerance", 1.0), ("double_tolerance", 2.0)):
        env = make_env(scale, args.rays)
        result["scenarios"][name] = {}
        for label, radius in (("init_pm_0.5mm", 0.5), ("travel_pm_0.8mm", 0.8)):
            vals, surfaces = scan(env, radius, args.grid)
            result["scenarios"][name][label] = {"axis_mm": vals.tolist(), "q_by_field": surfaces.tolist(), "summary": summarize(vals, surfaces)}
        env.close()
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    args.html.write_text(render_html(result), encoding="utf-8")
    print(f"wrote {args.output} and {args.html}")


def render_html(result: dict) -> str:
    """Create a self-contained heatmap viewer for the generated scan."""
    payload = json.dumps(result, ensure_ascii=False)
    return """<!doctype html><meta charset='utf-8'><title>MTF tolerance terrain</title>
<style>body{font:14px system-ui;margin:20px;color:#222}select{padding:5px}#plots{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:18px;margin-top:16px}svg{width:100%;border:1px solid #ddd}.cell{stroke:#fff;stroke-width:1}table{border-collapse:collapse;margin-top:18px}td,th{border:1px solid #ddd;padding:4px 7px;text-align:right}</style>
<h2>MTF terrain by tolerance</h2><label>Tolerance <select id='tol'></select></label> <label>Range <select id='range'></select></label><div id='plots'></div><table id='summary'></table>
<script>const D=__DATA__;const tol=document.querySelector('#tol'),range=document.querySelector('#range');Object.keys(D.scenarios).forEach(x=>tol.add(new Option(x,x)));Object.keys(D.scenarios[Object.keys(D.scenarios)[0]]).forEach(x=>range.add(new Option(x,x)));function draw(){const S=D.scenarios[tol.value][range.value],a=S.axis_mm,plots=document.querySelector('#plots');plots.innerHTML='';let mn=Infinity,mx=-Infinity;S.q_by_field.flat(2).forEach(v=>{mn=Math.min(mn,v);mx=Math.max(mx,v)});S.q_by_field.forEach((q,fi)=>{const n=400,p=44,size=300,c=size/q.length,svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.setAttribute('viewBox','0 0 '+n+' '+(n+45));const t=document.createElementNS(svg.namespaceURI,'text');t.setAttribute('x',n/2);t.setAttribute('y',18);t.setAttribute('text-anchor','middle');t.textContent='Field '+fi+' ('+D.fields_deg[fi].join(',')+' deg)';svg.append(t);q.forEach((row,y)=>row.forEach((v,x)=>{const r=document.createElementNS(svg.namespaceURI,'rect');r.setAttribute('x',p+x*c);r.setAttribute('y',25+(q.length-1-y)*c);r.setAttribute('width',c);r.setAttribute('height',c);r.setAttribute('class','cell');r.setAttribute('fill','hsl('+((v-mn)/(mx-mn)*210)+' 75% 45%)');r.title=a[x].toFixed(2)+', '+a[y].toFixed(2)+' mm: q='+v.toFixed(4);svg.append(r)}));plots.append(svg)});document.querySelector('#summary').innerHTML='<tr><th>Field</th><th>Peak q</th><th>Peak dx,dy</th><th>Center-edge drop</th></tr>'+S.summary.map(x=>'<tr><td>'+x.field_index+'</td><td>'+x.peak_q.toFixed(4)+'</td><td>'+x.peak_dx_mm.toFixed(3)+', '+x.peak_dy_mm.toFixed(3)+'</td><td>'+x.center_to_edge_drop.toFixed(4)+'</td></tr>').join('')}tol.onchange=range.onchange=draw;draw();</script>""".replace("__DATA__", payload)


if __name__ == "__main__":
    main()
