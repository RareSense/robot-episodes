r"""Pass/fail gates for the H1-2 tea episode (PhysX). The gates only read the simulation: no writes, no forces.

    import gates, props
    hook = gates.contact_pre(objects=gates.scene_sensors(), extra_paths=list(props.prop_paths().values()))   # room bodies, props, hands, feet
    s = props.make_sim(app, pre=[hook], xy=...)
    g = gates.Gates(s, still={"rack": "object_0310", "mug_0317": "object_0317"}, contact_paths=hook.paths, spill=my_ml_callback)
    g.attach()                     # every s.step() (also inside s.go and s.reach) is now checked
    g.baseline()                   # after settling: reference poses for the "not meant to move" gate
    g.hold("object_0316", "L_hand_base_link")     # when the grip is closed; g.release("object_0316") when it opens
    g.allow_move("object_0316")    # this object is now meant to move
    res = g.report(r"...\gates.json")

Each gate gives PASS, FAIL or NOT MEASURED plus its number. PASS needs real samples: no samples, or a failed read, is NOT MEASURED.
overall is PASS only if every gate is PASS; FAIL if any gate fails; otherwise INCOMPLETE.

What each gate reads
  fall          pelvis z from the articulation root; fail below 0.6 m.
  torque        |get_dof_projected_joint_forces| against URDF effort limits (parsed here, not taken from PhysX). Also records
                get_dof_actuation_forces and checks that get_dof_max_forces matches the URDF. The projected force is the whole joint
                force: foot impacts give 1-4 step spikes far above what the clamped drive can deliver (torque_diag.py). Default is strict:
                any sample over the limit fails. torque_spike_steps=N instead tolerates runs of at most N steps, if the motor estimate
                (clipped PD + actuation force) stays within limits. attach() after the spawn landing, or the drop counts.
  penetration   deepest negative separation from the rigid contact view's raw contact data, over the sensor bodies only.
                A pair with neither body in the sensor list is not seen. Sleeping bodies report no contacts.
  slip          held object's position in the holding link's frame, change since hold(); needs hold() to have been called.
  moved         listed objects' distance from their baseline().
  spill         the callback's ml; None or NaN from it counts as no sample.
"""
import json
import math
import xml.etree.ElementTree as ET

import numpy as np
from pxr import PhysxSchema, Usd, UsdPhysics

FALL_Z, PEN_MM, SLIP_MM, MOVE_MM, SPILL_ML = 0.6, 2.0, 5.0, 5.0, 3.0
HAND_AND_FEET = ("L_", "R_", "left_ankle", "right_ankle", "left_wrist", "right_wrist")
OBJECTS = ["object_0310"] + [f"object_{i:04d}" for i in range(311, 322)] + ["object_0299", "object_0305", "object_0306", "object_0309", "object_0295"]


def scene_sensors():
    return list(OBJECTS)


def _find(stage, name):
    """A body given as object_NNNN or as a full path."""
    if name.startswith("/"):
        return name
    for p in stage.Traverse():
        if p.GetName() == name and p.GetTypeName() == "Xform":
            return p.GetPath().pathString
    raise KeyError(name)


def contact_pre(robot_links=HAND_AND_FEET, objects=(), extra_paths=()):
    """pre-physics hook: PhysxContactReportAPI (threshold 0) on the chosen bodies, session layer."""
    def f(st):
        done = []
        for name in list(objects) + list(extra_paths):
            p = st.GetPrimAtPath(_find(st, name))
            if p and p.HasAPI(UsdPhysics.RigidBodyAPI):
                PhysxSchema.PhysxContactReportAPI.Apply(p).CreateThresholdAttr(0.0)
                done.append(p.GetPath().pathString)
        root = st.GetPrimAtPath("/World/h1_2")
        for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if p.HasAPI(UsdPhysics.RigidBodyAPI) and p.GetName().startswith(tuple(robot_links)):
                if p.IsInstanceProxy():
                    print("contact_pre: cannot author on instance proxy", p.GetPath(), flush=True)
                    continue
                PhysxSchema.PhysxContactReportAPI.Apply(p).CreateThresholdAttr(0.0)
                done.append(p.GetPath().pathString)
        f.paths = done
        print("contact_pre: report API on", len(done), "bodies", flush=True)
    f.paths = []
    return f


class Gates:
    def __init__(self, sim, still=None, contact_bodies=None, contact_paths=None, spill=None, every=1, spill_every=60, limit_scale=1.0,
                 max_contacts=4096, name="episode", torque_spike_steps=0):
        self.s, self.every, self.spill_cb, self.spill_every, self.name = sim, every, spill, spill_every, name
        st = sim.stage
        self.k = 0
        self.steps = 0
        self.enabled = True
        self.errors = {}
        self.fall = {"min_z": math.inf, "n": 0, "t": None, "max_tilt_deg": 0., "invalid": False}
        # torque
        import h12_physx as H
        self.H = H
        ur = ET.parse(H.URDF).getroot()
        eff = {j.get("name"): float(j.find("limit").get("effort")) for j in ur.findall("joint") if j.find("limit") is not None}
        self.dof = list(sim.names)
        self.lim = np.array([eff.get(x, np.nan) for x in self.dof], float) * limit_scale
        self.limit_scale = limit_scale
        self.no_limit = [x for x, v in zip(self.dof, self.lim) if not (v > 0)]
        self.tq = {"worst_ratio": 0.0, "n": 0, "joint": None, "nm": None, "t": None, "act_worst_nm": 0.0, "per_joint": np.zeros(len(self.dof)),
                   "steps_over": 0, "longest_run": 0, "motor_ratio": 0.0, "motor_joint": None}
        self.run_over = np.zeros(len(self.dof), int)
        self.torque_spike_steps = torque_spike_steps
        mf = np.asarray(sim.art.get_dof_max_forces())[0].astype(float)
        ok = self.lim > 0
        self.max_force_mismatch = [(self.dof[i], float(mf[i]), float(self.lim[i] / limit_scale)) for i in np.nonzero(ok)[0]
                                   if abs(mf[i] - self.lim[i] / limit_scale) > 1e-3 * max(1.0, abs(mf[i]))]
        # moved
        self.still_paths = {k: _find(st, v) for k, v in (still or {}).items()}
        self.still = {k: sim.sv.create_rigid_body_view(v) for k, v in self.still_paths.items()}
        self.base = {}
        self.moved = {k: {"mm": 0.0, "t": None, "n": 0, "released": False} for k in self.still}
        # penetration
        paths = [_find(st, x) for x in (contact_bodies or [])] + list(contact_paths or [])
        self.pen = {"mm": 0.0, "n_reads": 0, "n_contacts": 0, "t": None, "pair": None, "sensors": len(paths)}
        self.cv = None
        if paths:
            try:
                self.cv = sim.sv.create_rigid_contact_view(paths, max_contact_data_count=max_contacts)
                self.sensor_paths = list(np.ravel(self.cv.sensor_paths))
            except Exception as e:
                self.errors["penetration"] = f"view: {e!r}"
        # slip
        self.held = {}
        self.slip = {}
        # spill
        self.sp = {"ml": None, "n": 0}
        self._orig = None

    def attach(self):
        """Check every physics step, including those inside s.go and s.reach."""
        s = self.s
        if self._orig is None:
            self._orig = s.step

            def step():
                self._orig()
                self.tick()
            s.step = step
        return self

    def detach(self):
        if self._orig is not None:
            del self.s.step
            self._orig = None

    def baseline(self):
        for k, p in self._still_positions().items():
            self.base[k] = p.copy()

    def _still_positions(self):
        """Batch watched bodies without reducing the 600 Hz checking rate."""
        signature = tuple(self.still_paths.items())
        if not signature:
            return {}
        if getattr(self, '_still_signature', None) != signature:
            self._still_batch = self.s.sv.create_rigid_body_view(list(dict.fromkeys(self.still_paths.values())))
            paths = list(map(str, self._still_batch.prim_paths))
            self._still_order = {k: paths.index(p) for k, p in signature}
            self._still_signature = signature
        poses = self._still_batch.get_transforms()
        return {k: poses[i, :3].astype(float) for k, i in self._still_order.items()}

    def allow_move(self, key):
        for label, path in self.still_paths.items():
            if key in (label, path, path.rsplit('/', 1)[-1]):
                self.moved[label]["released"] = True

    def watch_still(self, obj):
        """Begin a new stationary phase after an intentional transport/release.

        Retain earlier maximum motion errors; only the reference pose changes.
        """
        path = _find(self.s.stage, obj)
        key = next((k for k, p in self.still_paths.items() if p == path), obj)
        if key not in self.still:
            self.still_paths[key] = path
            self.still[key] = self.s.sv.create_rigid_body_view(path)
            self.moved[key] = {"mm": 0., "t": None, "n": 0, "released": False}
        self.base[key] = self.still[key].get_transforms()[0, :3].astype(float).copy()
        self.moved[key]["released"] = False

    def _link_pose(self, link):
        i = self.s.links.index(link)
        return self.s.link_poses()[i].copy()

    def hold(self, obj, link):
        v = self.s.sv.create_rigid_body_view(_find(self.s.stage, obj))
        T = self._link_pose(link)
        p = v.get_transforms()[0].astype(float)[:3]
        ref = self.H.qrot(self.H.qinv(T[3:7]), p - T[:3])
        self.held[obj] = {"view": v, "link": link, "ref": ref}
        self.slip.setdefault(obj, {"mm": 0.0, "n": 0, "t": None})

    def release(self, obj):
        self.held.pop(obj, None)

    def _read(self, key, f):
        try:
            f()
        except Exception as e:
            self.errors.setdefault(key, repr(e))

    def pause(self):
        """Stop judging without unhooking (e.g. another Gates instance takes over a later phase)."""
        self.enabled = False

    def tick(self):
        if not self.enabled:
            return
        self.k += 1
        if self.k % self.every:
            return
        self.steps += 1
        s = self.s
        t = float(s.t)
        self._read("fall", lambda: self._fall(t))
        self._read("torque", lambda: self._torque(t))
        if self.cv is not None:
            self._read("penetration", lambda: self._pen(t))
        self._read("moved", lambda: self._moved(t))
        if self.held:
            self._read("slip", lambda: self._slip(t))
        if self.spill_cb is not None and self.k % self.spill_every == 0:
            self._read("spill", self._spill)

    def _fall(self, t):
        p, q = self.s.pose()
        z = float(p[2])
        if not np.isfinite(np.r_[p, q]).all():
            self.fall["invalid"] = True
            raise ValueError("non-finite root pose")
        up = self.H.qrot(q, np.array([0., 0., 1.]))
        self.fall["max_tilt_deg"] = max(self.fall["max_tilt_deg"], float(np.degrees(np.arccos(np.clip(up[2], -1, 1)))))
        if math.isfinite(z):
            self.fall["n"] += 1
            if z < self.fall["min_z"]:
                self.fall["min_z"], self.fall["t"] = z, t

    def _torque(self, t):
        a = self.s.art
        tau = np.abs(np.asarray(a.get_dof_projected_joint_forces())[0].astype(float))
        act = np.abs(np.asarray(a.get_dof_actuation_forces())[0].astype(float))
        if not np.all(np.isfinite(tau)):
            return
        self.tq["n"] += 1
        pos = self.lim > 0
        over = pos & (tau > np.where(pos, self.lim, np.inf))
        self.run_over = np.where(over, self.run_over + 1, 0)
        self.tq["steps_over"] += int(over.any())
        self.tq["longest_run"] = max(self.tq["longest_run"], int(self.run_over.max()))
        # motor estimate: the drive's PD torque clipped at its limit (PhysX clamps it) plus any feed-forward actuation force
        q, v = np.asarray(a.get_dof_positions())[0].astype(float), np.asarray(a.get_dof_velocities())[0].astype(float)
        tgt = np.asarray(a.get_dof_position_targets())[0].astype(float)
        kp, kd = np.asarray(a.get_dof_stiffnesses())[0].astype(float), np.asarray(a.get_dof_dampings())[0].astype(float)
        mf = np.asarray(a.get_dof_max_forces())[0].astype(float)
        sact = np.asarray(a.get_dof_actuation_forces())[0].astype(float)
        motor = np.abs(np.clip(kp * (tgt - q) - kd * v, -mf, mf) + sact)
        mr = np.where(pos, motor / np.where(pos, self.lim, 1.0), 0.0)
        j = int(np.argmax(mr))
        if mr[j] > self.tq["motor_ratio"]:
            self.tq.update(motor_ratio=float(mr[j]), motor_joint=self.dof[j])
        self.tq["act_worst_nm"] = max(self.tq["act_worst_nm"], float(np.nanmax(act)))
        self.tq["per_joint"] = np.maximum(self.tq["per_joint"], tau)
        r = np.where(self.lim > 0, tau / np.where(self.lim > 0, self.lim, 1.0), 0.0)
        i = int(np.argmax(r))
        if r[i] > self.tq["worst_ratio"]:
            self.tq.update(worst_ratio=float(r[i]), joint=self.dof[i], nm=float(tau[i]), t=t)

    def _pen(self, t):
        f, pt, nrm, sep, cnt, start, other = self.cv.get_raw_contact_data(self.H.DT)
        sep = np.asarray(sep).reshape(-1)
        cnt = np.asarray(cnt).reshape(-1)
        start = np.asarray(start).reshape(-1)
        self.pen["n_reads"] += 1
        n = int(cnt.sum())
        if n == 0:
            return
        self.pen["n_contacts"] += n
        for i in np.nonzero(cnt)[0]:
            a, b = int(start[i]), int(start[i] + cnt[i])
            seg = sep[a:b]
            j = int(np.argmin(seg))
            d = -float(seg[j]) * 1000
            if d > self.pen["mm"]:
                who = None
                try:
                    who = self.cv.get_other_actor_paths_from_ids(np.asarray(other).reshape(-1)[a + j:a + j + 1])
                    who = str(who[0]) if len(who) else None
                except Exception:
                    pass
                self.pen.update(mm=d, t=t, pair=[str(self.sensor_paths[i]), who])

    def _moved(self, t):
        positions = self._still_positions()
        for k, v in self.still.items():
            m = self.moved[k]
            if m["released"] or k not in self.base:
                continue
            d = float(np.linalg.norm(positions[k] - self.base[k])) * 1000
            if math.isfinite(d):
                m["n"] += 1
                if d > m["mm"]:
                    m["mm"], m["t"] = d, t

    def _slip(self, t):
        for k, h in self.held.items():
            T = self._link_pose(h["link"])
            p = h["view"].get_transforms()[0].astype(float)[:3]
            d = float(np.linalg.norm(self.H.qrot(self.H.qinv(T[3:7]), p - T[:3]) - h["ref"])) * 1000
            e = self.slip[k]
            if math.isfinite(d):
                e["n"] += 1
                if d > e["mm"]:
                    e["mm"], e["t"] = d, t

    def _spill(self):
        v = self.spill_cb()
        if v is None or not math.isfinite(float(v)):
            return
        self.sp["n"] += 1
        self.sp["ml"] = max(float(v), self.sp["ml"] if self.sp["ml"] is not None else -math.inf)

    def final_spill(self):
        if self.spill_cb is not None:
            self._read("spill", self._spill)

    def judge(self):
        NM, P, F = "NOT MEASURED", "PASS", "FAIL"
        err = self.errors
        out = {}

        def g(key, ok, value, unit, limit, n, **more):
            if key in err:
                st, note = NM, "read error: " + err[key]
            elif n == 0 or value is None:
                st, note = NM, "no samples"
            else:
                st, note = (P if ok else F), ""
            out[key] = dict(status=st, value=None if st == NM else value, unit=unit, limit=limit, samples=n, note=note, **more)

        z = self.fall["min_z"] if self.fall["n"] else None
        g("fall", z is not None and z >= FALL_Z and self.fall["max_tilt_deg"] <= 55 and not self.fall["invalid"], z,
          "m min pelvis z", f">= {FALL_Z}; pelvis tilt <= 55 degrees", self.fall["n"], at_t=self.fall["t"], max_tilt_deg=self.fall["max_tilt_deg"])
        r = self.tq["worst_ratio"] if self.tq["n"] else None
        top = sorted(zip(self.dof, self.tq["per_joint"].tolist(), (self.lim / self.limit_scale).tolist()), key=lambda x: -(x[1] / x[2] if x[2] > 0 else 0))[:5]
        spike = self.torque_spike_steps
        tq_ok = r is not None and (r <= 1.0 if spike == 0 else self.tq["longest_run"] <= spike and self.tq["motor_ratio"] <= 1.0)
        g("torque", tq_ok, r, "worst |tau|/limit", "<= 1.0" if spike == 0 else f"over limit for at most {spike} steps in a row, motor estimate <= 1.0",
          self.tq["n"], joint=self.tq["joint"], nm=self.tq["nm"], at_t=self.tq["t"], steps_over=self.tq["steps_over"], longest_run_steps=self.tq["longest_run"],
          motor_estimate_ratio=self.tq["motor_ratio"], motor_estimate_joint=self.tq["motor_joint"],
          limit_scale=self.limit_scale, joints_without_urdf_limit=self.no_limit, max_force_mismatch=self.max_force_mismatch[:6],
          actuation_force_worst_nm=self.tq["act_worst_nm"], top5=[[a, round(b, 3), c] for a, b, c in top])
        if self.tq["n"] and self.tq["per_joint"].max() == 0:
            out["torque"].update(status=NM, value=None, note="projected joint forces all exactly zero: not a real reading")
        if out["torque"]["status"] != NM:
            out["torque"]["note"] = (f"{self.tq['steps_over']} steps over, longest run {self.tq['longest_run']}, motor estimate {self.tq['motor_ratio']:.3f} "
                                     f"({self.tq['motor_joint']})")
        if self.no_limit and out["torque"]["status"] == P:
            out["torque"]["note"] = f"{len(self.no_limit)} joints have no URDF limit and were not judged"
        pm = self.pen["mm"] if self.pen["n_contacts"] else None
        g("penetration", pm is not None and pm <= PEN_MM, pm, "mm deepest", f"<= {PEN_MM}", self.pen["n_contacts"], at_t=self.pen["t"], pair=self.pen["pair"],
          sensors=self.pen["sensors"], reads=self.pen["n_reads"])
        if self.cv is None and "penetration" not in err:
            out["penetration"]["note"] = "no contact bodies given"
        sl = max((v["mm"] for v in self.slip.values()), default=None)
        ns = sum(v["n"] for v in self.slip.values())
        g("slip", sl is not None and sl <= SLIP_MM, sl, "mm in hand frame", f"<= {SLIP_MM}", ns, per_object={k: round(v["mm"], 3) for k, v in self.slip.items()})
        watched = {k: v for k, v in self.moved.items()}
        mv = max((v["mm"] for v in watched.values() if v["n"]), default=None)
        nm_ = sum(v["n"] for v in watched.values())
        worst = max(watched, key=lambda k: watched[k]["mm"]) if watched else None
        g("moved", mv is not None and mv < MOVE_MM, mv, "mm worst object", f"< {MOVE_MM}", nm_, worst=worst, per_object={k: round(v["mm"], 3) for k, v in watched.items()},
          never_sampled=[k for k, v in watched.items() if not v["n"]], released=[k for k, v in watched.items() if v["released"]])
        g("spill", self.sp["ml"] is not None and self.sp["ml"] <= SPILL_ML, self.sp["ml"], "ml", f"<= {SPILL_ML}", self.sp["n"])
        sts = [v["status"] for v in out.values()]
        overall = "FAIL" if F in sts else ("INCOMPLETE" if NM in sts else P)
        return {"name": self.name, "overall": overall, "steps_checked": self.steps, "sim_time": float(self.s.t), "gates": out}

    def report(self, path=None):
        self.final_spill()
        r = self.judge()
        print(f"=== gates: {r['name']}  overall {r['overall']}  ({r['steps_checked']} steps checked)", flush=True)
        for k, v in r["gates"].items():
            val = "-" if v["value"] is None else f"{v['value']:.3f}"
            print(f"  {k:12s} {v['status']:13s} {val} {v['unit']} (limit {v['limit']}, {v['samples']} samples) {v['note']}", flush=True)
        if path:
            json.dump(r, open(path, "w"), indent=1, default=float)
        return r
