#!/usr/bin/env python3
"""
generate_synthetic_corpus.py
============================
Programmatically generates a 50,000-record synthetic corpus specialized for:
  * mock structural building codes (IBC/ASCE/ACI/AISC/Eurocode style provisions)
  * seismic load differential equations (SDOF/MDOF, response spectra, damping)
  * mock Revit API documentation (Autodesk.Revit.DB automation)
  * mock SAP2000 API documentation (CSI OAPI automation)

Also emits:
  * data/structural_nodes.json  - 8,000 structural node feature vectors + failure
                                  labels (ground truth for the ML failure-probability
                                  classifier in rag_engine.py)
  * data/queries.json           - 500 benchmark queries with ground-truth relevant
                                  record ids (cluster based) for precision/recall.

Every record belongs to a topic "cluster" (4-8 records sharing one engineering
topic phrase that appears verbatim in each record's text). The benchmark measures
recall over clusters, which makes precision/recall well defined.

Deterministic: fixed seed. Output: data/corpus.json (UTF-8).
"""
import json
import math
import random
import time
from pathlib import Path

SEED = 42
TOTAL_RECORDS = 50_000
N_CODE = 20_000
N_SEISMIC = 8_000
N_REVIT = 12_000
N_SAP = 10_000
assert N_CODE + N_SEISMIC + N_REVIT + N_SAP == TOTAL_RECORDS

N_NODES = 8_000
N_QUERIES = 500
CLUSTER_MIN, CLUSTER_MAX = 4, 8

RNG = random.Random(SEED)
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Topic vocabulary: (subject, qualifier, context) triples must be unique per
# cluster so that benchmark queries have a well-defined relevant set.
# --------------------------------------------------------------------------

CODE_SUBJECTS = [
    "load combination", "seismic base shear", "wind pressure", "story drift limit",
    "reinforcement detailing", "shear wall design", "steel moment connection",
    "composite beam design", "foundation bearing pressure", "snow load",
    "concrete column confinement", "timber diaphragm", "masonry anchorage",
    "torsional irregularity", "redundancy factor", "overstrength factor",
    "diaphragm chord force", "pile cap design", "continuity plates", "panel zone shear",
]
CODE_QUALIFIERS = [
    "design requirements", "strength provisions", "detailing rules", "load factors",
    "capacity checks", "serviceability limits", "quality control", "inspection requirements",
    "material specifications", "analysis procedure",
]
CODE_CONTEXTS = [
    "seismic design category D", "seismic design category C", "Risk Category IV hospitals",
    "high seismic zones", "coastal wind regions", "cold regions with heavy snowfall",
    "multi-story moment frames", "low-rise shear wall buildings", "parking structures",
    "tall buildings over 60 meters", "industrial facilities", "precast plants",
    "retrofit projects", "essential facilities", "open buildings",
]

SEISMIC_SUBJECTS = [
    "SDOF equation of motion", "response spectrum ordinate", "modal combination rule",
    "Rayleigh damping model", "base shear coefficient", "displacement ductility demand",
    "Newmark integration scheme", "pushover target displacement",
    "capacity spectrum method", "soil structure interaction correction",
    "tuned mass damper response", "inelastic demand modification",
    "vertical ground motion component", "near fault velocity pulse",
    "energy based design formula",
]
SEISMIC_QUALIFIERS = [
    "derivation", "numerical solution", "parameter calibration", "validation example",
    "sensitivity analysis", "design application", "code alignment", "worked calculation",
]
SEISMIC_CONTEXTS = [
    "firm soil site class D", "soft clay site class E", "rock site class B",
    "stiff soil site class C", "short period structures", "medium period structures",
    "long period high rise", "base isolated systems", "damped braced frames",
    "light single story buildings", "irregular multi story frames",
    "tall chimney structures", "liquid storage tanks", "bridge piers", "machine foundations",
]

REVIT_SUBJECTS = [
    "family parameter creation", "wall placement", "level creation",
    "element collection filter", "transaction handling", "rebar modeling",
    "view creation", "DWG export", "unit conversion", "beam family placement",
    "floor sketch editing", "parameter set read write", "element deletion undo",
    "schedule field creation", "adaptive component update", "family type catalog",
    "shared parameter binding", "workset checkout",
]
REVIT_QUALIFIERS = [
    "API method", "C# example", "Python example", "error handling pattern",
    "performance note", "best practice", "transaction guidance", "override behavior",
]
REVIT_CONTEXTS = [
    "structural framing workflow", "architectural template", "BIM 360 cloud model",
    "linked model workflow", "family editor session", "construction documentation set",
    "analysis model export", "clash detection review", "as built documentation",
    "conceptual massing study",
]

SAP_SUBJECTS = [
    "frame object assignment", "load pattern definition", "material property definition",
    "analysis run", "result extraction", "load combination creation",
    "frame section property", "support constraint assignment", "link element definition",
    "steel design check", "concrete design check", "modal analysis setup",
    "response spectrum function", "diaphragm assignment", "auto seismic load",
    "tendon object modeling", "staged construction sequence", "grid line generation",
]
SAP_QUALIFIERS = [
    "API function", "return code handling", "Python comtypes example", "CScript example",
    "units note", "performance tip", "reset procedure", "refresh view note",
]
SAP_CONTEXTS = [
    "steel gravity frame model", "concrete shear wall building", "composite floor system",
    "long span truss roof", "bridge girder model", "tower structure model",
    "industrial pipe rack", "stadium roof structure", "parking garage model",
    "seismic retrofit model",
]

STANDARDS = ["IBC-2024", "ASCE 7-22", "ACI 318-19", "AISC 360-22", "Eurocode 2",
             "Eurocode 8", "SBC-301", "CSA A23.3"]
MATERIALS = ["A992 Gr.50 steel", "A36 steel", "C30/37 concrete", "C40/50 concrete",
             "GL24h glulam", "CFS 600S162-43", "Type N mortar masonry", "Fy=420 rebar"]
SECTIONS_W = ["W24X62", "W18X35", "W14X22", "W21X50", "HSS8x8x1/2", "HSS6x6x3/8",
              "W12X40", "W16X26", "MC9x23.9", "WT7X15"]
ZONES = ["zone 2B", "zone 3", "zone 4", "SDC A", "SDC B", "SDC C", "SDC D", "SDC E"]


def draw_unique_triples(subjects, qualifiers, contexts, count):
    """Draw up to `count` unique (subject, qualifier, context) triples.

    If the vocabulary cross-product is smaller than `count`, all unique
    triples are returned (clusters then average slightly larger)."""
    seen, out, tries, max_tries = set(), [], 0, 2_000_000
    while len(out) < count and tries < max_tries:
        tries += 1
        t = (RNG.choice(subjects), RNG.choice(qualifiers), RNG.choice(contexts))
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def cap(s):
    return s[0].upper() + s[1:] if s else s


# --------------------------------------------------------------------------
# ML failure-probability node generator (shared by nodes file and code meta)
# --------------------------------------------------------------------------

def make_node(rng, prefix, idx):
    """Synthetic structural node: features + failure label via noisy logistic model."""
    dcr = round(rng.uniform(0.15, 2.10), 3)            # demand/capacity ratio
    drift = round(rng.uniform(0.001, 0.045), 4)        # inter-story drift ratio
    axial = round(rng.uniform(0.02, 0.92), 3)          # axial load ratio
    duct = round(rng.uniform(0.8, 8.0), 2)             # curvature ductility supply
    corr = round(rng.uniform(0.0, 22.0), 2)            # corrosion section loss %
    zone = rng.choice([0.08, 0.12, 0.16, 0.20, 0.28, 0.36, 0.44, 0.52])  # zonal PGA
    slip = round(rng.uniform(0.0, 4.0), 2)             # connection slip mm
    dur = rng.choice([1.0, 1.15, 1.25, 0.9])           # load duration factor
    logit = (-7.4 + 4.6 * dcr + 58.0 * drift + 1.9 * axial
             - 0.55 * duct + 0.11 * corr + 3.2 * zone + 0.35 * slip
             - 0.8 * (dur - 1.0) + rng.gauss(0.0, 0.55))
    p = 1.0 / (1.0 + math.exp(-logit))
    label = 1 if rng.random() < p else 0
    feats = {
        "demand_capacity_ratio": dcr, "drift_ratio": drift, "axial_load_ratio": axial,
        "curvature_ductility": duct, "corrosion_pct": corr, "zone_pga": zone,
        "connection_slip_mm": slip, "load_duration_factor": dur,
    }
    return {"id": f"{prefix}-{idx:06d}", "features": feats, "failure_label": label}


# --------------------------------------------------------------------------
# Record content builders (each guarantees the cluster topic phrase appears
# verbatim in the text so BM25/dense retrieval is well posed)
# --------------------------------------------------------------------------

def code_content(topic, rng, i):
    subject, qualifier, context = topic
    std = RNG_STDS[i % len(RNG_STDS)] if (RNG_STDS := STANDARDS) else STANDARDS[0]
    section = f"{rng.randint(1, 33)}.{rng.randint(1, 22)}.{rng.randint(1, 99)}"
    mat = rng.choice(MATERIALS)
    zone = rng.choice(ZONES)
    phi = round(rng.uniform(0.65, 0.95), 2)
    fs = rng.choice([0.75, 0.8, 0.9, 1.0, 1.2, 1.6])
    load = round(rng.uniform(5.0, 180.0), 1)
    variant = rng.randint(0, 4)
    if variant == 0:
        body = (f"Provisions for {subject} and {qualifier} apply to {context}. "
                f"Per {std} section {section}, the nominal strength shall be multiplied "
                f"by the resistance factor phi={phi} when {mat} members are evaluated "
                f"in {zone}. The load factor modifier is {fs} and the reference demand "
                f"is {load} kN. Connections shall develop the required strength without "
                f"exceeding the amplified seismic load.")
    elif variant == 1:
        body = (f"{cap(std)} requires that {qualifier} for {subject} in {context} be "
                f"verified with load factor {fs} and phi={phi}. Tabulated values for "
                f"{mat} are permitted only when the compactness limits of section "
                f"{section} are met. Where {zone} governs, the redundancy factor "
                f"amplifies member demands and deflections shall not exceed L/240 "
                f"under service loads.")
    elif variant == 2:
        body = (f" commentary for {context} addresses {subject} {qualifier}: the "
                f"design action equals {load} kN times the {fs} overstrength factor, "
                f"and capacity is computed from {mat} with phi={phi}. Section "
                f"{section} lists weld and bolt requirements, including minimum "
                f"notch toughness of 20 ft-lbf at -20 F for seismic resisting systems "
                f"assigned to {zone}.")
    elif variant == 3:
        body = (f"For {context}, {qualifier} governing {subject} follow {std} "
                f"{section}. The strength reduction factor phi={phi} applies to "
                f"{mat}; the overstrength modifier {fs} applies to collectors and "
                f"discontinuous elements. Reference load {load} kN shall be combined "
                f"per the basic seismic load combinations, and detailing in {zone} "
                f"requires special inspection.")
    else:
        body = (f"{cap(qualifier)} for {subject} in {context}: {std} section {section} "
                f"permits elastic analysis with R=8, Cd=5.5 and Omega0=2.5 unless the "
                f"exception for {mat} applies. Check the demand {load} kN against "
                f"phi={phi} capacity, apply the {fs} factor to accidental torsion, "
                f"and document drift sensitivity for {zone}.")
    tail = (f" Cross-reference: related {rng.choice(STANDARDS)} clauses cover "
            f"{rng.choice(CODE_SUBJECTS)} in the same {context} category.") \
        if rng.random() < 0.35 else ""
    return body + tail


def seismic_content(topic, rng, i):
    subject, qualifier, context = topic
    m = round(rng.uniform(8.0, 900.0), 1)        # kg
    k = round(rng.uniform(2e5, 9e7), 0)          # N/m
    wn = math.sqrt(k / m)
    T = 2 * math.pi / wn
    zeta = round(rng.uniform(0.02, 0.15), 3)
    ag = round(rng.uniform(0.08, 0.62), 3)       # g
    variant = rng.randint(0, 4)
    if variant == 0:
        body = (f"Differential equation notes for {subject}, {qualifier}, {context}: "
                f"m*u''(t) + c*u'(t) + k*u(t) = -m*ag(t), with m={m} kg, "
                f"k={k:.3e} N/m, c={zeta * 2 * m * wn:.1f} N*s/m. The undamped natural "
                f"frequency wn=sqrt(k/m)={wn:.3f} rad/s gives period T={T:.3f} s; "
                f"ground acceleration ag(t) peaks at {ag} g.")
    elif variant == 1:
        body = (f"For {context}, the {qualifier} of the {subject} proceeds from "
                f"wn={wn:.3f} rad/s (T={T:.3f} s) and damping zeta={zeta:.3f}. The "
                f"frequency equation m*u'' + k*u = -m*ag(t) with m={m} kg and "
                f"k={k:.3e} N/m admits u(t)=C1*sin(wn*t)+C2*cos(wn*t)+particular. "
                f"Spectral amplification Sa exceeds {ag} g at resonance.")
    elif variant == 2:
        body = (f"{cap(qualifier)} of the {subject} for {context}: assume Rayleigh "
                f"damping C=a0*M+a1*K with zeta={zeta:.3f} at frequencies "
                f"{wn:.3f} and {2.6 * wn:.3f} rad/s. Solving m*u''(t)+c*u'(t)+k*u(t)="
                f"-m*ag(t) for m={m} kg, k={k:.3e} N/m yields peak displacement "
                f"Sd=Sa/wn^2 and base shear V=m*Sa; design ag={ag} g.")
    elif variant == 3:
        body = (f"Worked calculation - {subject} ({qualifier}) for {context}. With "
                f"m={m} kg, k={k:.3e} N/m, wn=sqrt(k/m)={wn:.3f} rad/s, T={T:.3f} s, "
                f"the EOM m*u''+c*u'+k*u=-m*ag(t) is integrated by the Newmark scheme "
                f"(beta=1/4, gamma=1/2) under ag(t)={ag} g pulse. Damping zeta="
                f"{zeta:.3f}; ductility demand mu<=4.0 assumed.")
    else:
        body = (f"{cap(context)} case study for the {subject}: {qualifier} shows that "
                f"for T={T:.3f} s (wn={wn:.3f} rad/s), zeta={zeta:.3f}, the equation "
                f"m*u''(t)+c*u'(t)+k*u(t)=-m*ag(t) with m={m} kg and k={k:.3e} N/m "
                f"produces spectral ordinate Sa={ag} g. SRSS modal combination of the "
                f"first three modes governs; check drift and P-Delta effects.")
    return body


REVIT_CLASSES = ["Document", "ElementId", "FilteredElementCollector", "Transaction",
                 "FamilyManager", "FamilySymbol", "Wall", "Level", "Grid",
                 "IndependentTag", "Parameter", "UnitUtils", "View3D", "Rebar",
                 "FamilyInstance", "Group", "ScheduleSheet", "ExportDWGOptions"]


def revit_content(topic, rng, i):
    subject, qualifier, context = topic
    cls = rng.choice(REVIT_CLASSES)
    method = rng.choice(["Create", "NewFamilyInstance", "get_Parameter",
                         "SetElementId", "Collect", "Commit", "Rollback",
                         "Delete", "Regenerate", "Move", "Rotate", "Copy"])
    ver = rng.choice(["Revit 2023", "Revit 2024", "Revit 2025"])
    variant = rng.randint(0, 3)
    if variant == 0:
        body = (f"Revit {ver} API reference for {subject}: {qualifier} for {context}. "
                f"Namespace Autodesk.Revit.DB.{cls}.{method}() returns an ElementId "
                f"or a filtered Element collection. Wrap all model changes inside a "
                f"Transaction: tx.Start('edit'); doc.Regenerate(); tx.Commit(). "
                f"InvalidElementException is thrown when the id no longer exists.")
    elif variant == 1:
        body = (f"{cap(qualifier)} - Revit {ver} automation of {subject} in {context}. "
                f"Use {cls}.{method}() after doc.Regenerate(); obtain parameters via "
                f"element.get_Parameter(BuiltInParameter.STRUCTURAL_BEND_DIR_NUM) and "
                f"unit conversion through UnitUtils.ConvertToInternalUnits(value, "
                f"UnitTypeId.Millimeters). Commit exactly one Transaction per undo step.")
    elif variant == 2:
        body = (f"C# snippet for {subject} ({qualifier}, {context}): using(var tx = new "
                f"Transaction(doc, '{cls} edit')) {{ tx.Start(); var id = {cls}.{method}"
                f"(doc, refId); doc.Regenerate(); tx.Commit(); }} Filtering example: "
                f"new FilteredElementCollector(doc).OfClass(typeof({cls})).WhereElementIsNot"
                f"ElementType(); failures roll back via FailureHandlingOptions.")
    else:
        body = (f"Python (pyRevit/Revit Python Shell) pattern for {subject} - "
                f"{qualifier} in {context}: with Transaction(doc, 'rvt'): "
                f"ids = FilteredElementCollector(doc).OfCategory(BuiltInCategory."
                f"OST_StructuralFraming).ToElementIds(); then call {cls}.{method}() "
                f"and doc.Regenerate(). Note: {ver} enforces read-only access outside "
                f"an open Transaction; ids stay valid until undo.")
    return body


SAP_OBJECTS = ["SapModel.PropMaterial", "SapModel.PropFrame", "SapModel.FrameObj",
               "SapModel.AreaObj", "SapModel.PointObj", "SapModel.LoadPatterns",
               "SapModel.LoadCases.StaticLinear", "SapModel.RespCombo",
               "SapModel.Analyze", "SapModel.Results.JointDispl", "SapModel.Results.FrameForce",
               "SapModel.DesignSteel", "SapModel.DesignConcrete", "SapModel.Constraint",
               "SapModel.LinkObj", "SapModel.GroupDef", "cHelper"]


def sap_content(topic, rng, i):
    subject, qualifier, context = topic
    obj = rng.choice(SAP_OBJECTS)
    fn = rng.choice(["AddByCoord", "SetLoads", "Define", "RunAnalysis", "GetForces",
                     "SetAssignments", "AddByPoint", "Delete", "SetActive", "GetTable"])
    units = rng.choice(["kN_m_C", "kN_mm_C", "kgf_m_C", "N_mm_C"])
    ret = rng.choice([0, 0, 0, 1, 2])  # 0 = success convention
    variant = rng.randint(0, 3)
    if variant == 0:
        body = (f"SAP2000 OAPI reference for {subject}: {qualifier} for {context}. "
                f"Call {obj}.{fn}(...) with the model units set to {units}; the "
                f"function returns {ret} ({'success' if ret == 0 else 'nonzero - check eKey error code'}). "
                f"Typical call sequence: SapModel.InitializeNewModel(units); "
                f"SapModel.File.NewGrid(3, 6.0, 2, 4.0); then assign properties before "
                f"SapModel.Analyze.RunAnalysis().")
    elif variant == 1:
        body = (f"Python comtypes example - {subject} ({qualifier}, {context}): "
                f"SapModel = helper.GetObject('CSI.SAP2000.API.SapObject').SapModel; "
                f"SapModel.SetPresentUnits({units}); ret = {obj}.{fn}(...); assert "
                f"ret == {ret}. Refresh the view with SapModel.View.RefreshView() after "
                f"batch assignments; results tables need SapModel.Results.Setup.")
    elif variant == 2:
        body = (f"Units note for {subject} - {qualifier} in {context}: {obj}.{fn}() "
                f"interprets every numeric argument in the present units ({units}). "
                f"Return code {ret} indicates {'success' if ret == 0 else 'failure; inspect SapModel.GetLastErrors()'}. "
                f"Modal analysis setup requires SapModel.LoadCases.ModalHist or "
                f"SapModel.Analyze.SetCaseModal before extracting results.")
    else:
        body = (f"Return-code handling for {subject} ({qualifier}, {context}): each "
                f"{obj}.{fn}() call returns an integer status; loop over items, collect "
                f"statuses != 0, and call SapModel.View.RefreshView() once after the "
                f"batch. Present units {units}; RunAnalysis must be re-executed after "
                f"any stiffness change before result extraction.")
    return body


# --------------------------------------------------------------------------
# Cluster generation
# --------------------------------------------------------------------------

def build_clusters():
    """Return list of dicts: {category, topic, n_records}."""
    n_code_clusters = int(N_CODE / 6)
    n_seis_clusters = int(N_SEISMIC / 6)
    n_revit_clusters = int(N_REVIT / 6)
    n_sap_clusters = int(N_SAP / 6)
    plan = []
    for cat, subs, quals, ctxs, n in [
        ("building_code", CODE_SUBJECTS, CODE_QUALIFIERS, CODE_CONTEXTS, n_code_clusters),
        ("seismic_equation", SEISMIC_SUBJECTS, SEISMIC_QUALIFIERS, SEISMIC_CONTEXTS, n_seis_clusters),
        ("revit_api", REVIT_SUBJECTS, REVIT_QUALIFIERS, REVIT_CONTEXTS, n_revit_clusters),
        ("sap2000_api", SAP_SUBJECTS, SAP_QUALIFIERS, SAP_CONTEXTS, n_sap_clusters),
    ]:
        triples = draw_unique_triples(subs, quals, ctxs, n)
        for t in triples:
            plan.append({"category": cat, "topic": t,
                         "n": RNG.randint(CLUSTER_MIN, CLUSTER_MAX)})
    RNG.shuffle(plan)
    return plan


CAT_QUOTA = {"building_code": N_CODE, "seismic_equation": N_SEISMIC,
             "revit_api": N_REVIT, "sap2000_api": N_SAP}


def main():
    t0 = time.time()
    plan = build_clusters()

    records = []
    nodes = []
    cluster_index = []  # (cluster_id, category, subject, qualifier, context)
    counters = {"building_code": 0, "seismic_equation": 0, "revit_api": 0, "sap2000_api": 0}
    prefixes = {"building_code": "SBC", "seismic_equation": "SEQ",
                "revit_api": "RVT", "sap2000_api": "SAP"}
    builders = {"building_code": code_content, "seismic_equation": seismic_content,
                "revit_api": revit_content, "sap2000_api": sap_content}
    titles = {"building_code": "Structural Code Provision",
              "seismic_equation": "Seismic Differential Equation",
              "revit_api": "Revit API Documentation",
              "sap2000_api": "SAP2000 API Documentation"}

    def emit(cat, topic, cluster_id):
        counters[cat] += 1
        idx = counters[cat]
        rid = f"{prefixes[cat]}-{idx:06d}"
        content = builders[cat](topic, RNG, idx)
        rec = {
            "id": rid,
            "category": cat,
            "cluster_id": cluster_id,
            "title": f"{titles[cat]}: {topic[0]} ({topic[1]}) - {topic[2]}",
            "content": content,
            "meta": {"standard_prefix": prefixes[cat], "index": idx},
        }
        # ~50% of code records carry a structural node profile used by the
        # ML failure-probability classifier at retrieval time.
        if cat == "building_code" and RNG.random() < 0.5:
            node = make_node(RNG, "SNODE", len(nodes) + 1)
            rec["meta"]["node_features"] = node["features"]
            rec["meta"]["failure_label"] = node["failure_label"]
            nodes.append(node)
        records.append(rec)
        return rid

    cluster_id = 0
    cat_cluster_ids = {k: [] for k in CAT_QUOTA}
    for item in plan:
        cat = item["category"]
        topic = item["topic"]
        member_ids = [emit(cat, topic, cluster_id) for _ in range(item["n"])]
        cluster_index.append({
            "cluster_id": cluster_id, "category": cat,
            "subject": topic[0], "qualifier": topic[1], "context": topic[2],
            "member_ids": member_ids,
        })
        cat_cluster_ids[cat].append(cluster_id)
        cluster_id += 1

    # Top-up: vocabulary pools may cap the number of unique topic clusters,
    # so fill each category to its exact quota by appending records to
    # randomly chosen clusters of that category (cluster size grows slightly).
    for cat, quota in CAT_QUOTA.items():
        ids = cat_cluster_ids[cat]
        while counters[cat] < quota:
            cid = RNG.choice(ids)
            member_ids = cluster_index[cid]["member_ids"]
            member_ids.append(emit(cat, (cluster_index[cid]["subject"],
                                         cluster_index[cid]["qualifier"],
                                         cluster_index[cid]["context"]), cid))

    # Trim categories that overshot quota (keep every cluster at >= 3 members)
    removed = set()
    for cat, quota in CAT_QUOTA.items():
        ids = cat_cluster_ids[cat]
        while counters[cat] > quota:
            big = max(ids, key=lambda cid: len(cluster_index[cid]["member_ids"]))
            members = cluster_index[big]["member_ids"]
            if len(members) <= 3:
                break
            removed.add(members.pop())
            counters[cat] -= 1
    if removed:
        records = [r for r in records if r["id"] not in removed]

    # Extra standalone ML nodes (held-out evaluation pool for the classifier)
    while len(nodes) < N_NODES:
        nodes.append(make_node(RNG, "SNODE", len(nodes) + 1))

    # ---- Benchmark queries (500) -------------------------------------------
    by_cat = {}
    for c in cluster_index:
        by_cat.setdefault(c["category"], []).append(c)
    per_cat_quota = {
        "building_code": 200, "seismic_equation": 80,
        "revit_api": 120, "sap2000_api": 100,
    }
    wrappers = {
        "building_code": "code provisions requirements",
        "seismic_equation": "equation derivation formula",
        "revit_api": "Revit API reference method example",
        "sap2000_api": "SAP2000 API function example",
    }
    q_templates = {
        "building_code": ["What are the {q} for {s} in {c}?",
                          "Show code requirements: {s}, {q}, {c}.",
                          "{s} {q} applicable to {c}?"],
        "seismic_equation": ["Derive the {s} ({q}) for {c}.",
                             "Equation for {s} in {c}, {q} needed.",
                             "Give the {q} of the {s} for {c}."],
        "revit_api": ["Revit API for {s}: {q} for {c}?",
                      "How to automate {s} ({q}) in {c}?",
                      "Show the {q} covering {s} in {c}."],
        "sap2000_api": ["SAP2000 OAPI call for {s}: {q} in {c}.",
                        "How to script {s} ({q}) for {c}?",
                        "Find the {q} for {s} in {c}."],
    }
    queries = []
    qid = 0
    for cat, quota in per_cat_quota.items():
        pool = by_cat[cat]
        picks = RNG.sample(pool, min(quota, len(pool)))
        for c in picks:
            qid += 1
            tmpl = RNG.choice(q_templates[cat])
            qtext = tmpl.format(s=c["subject"], q=c["qualifier"], c=c["context"])
            qtext = f"{qtext} {wrappers[cat]}"
            queries.append({
                "qid": f"Q{qid:04d}",
                "query": qtext,
                "category": cat,
                "cluster_id": c["cluster_id"],
                "relevant_ids": c["member_ids"],
            })
    RNG.shuffle(queries)

    # ---- Write outputs ------------------------------------------------------
    corpus_path = DATA_DIR / "corpus.json"
    with open(corpus_path, "w", encoding="utf-8") as f:
        json.dump({"seed": SEED, "count": len(records), "records": records},
                  f, ensure_ascii=False)

    nodes_path = DATA_DIR / "structural_nodes.json"
    with open(nodes_path, "w", encoding="utf-8") as f:
        json.dump({"count": len(nodes), "nodes": nodes}, f)

    queries_path = DATA_DIR / "queries.json"
    with open(queries_path, "w", encoding="utf-8") as f:
        json.dump({"count": len(queries), "queries": queries}, f, ensure_ascii=False)

    # ---- Stats --------------------------------------------------------------
    cat_counts = {k: v for k, v in counters.items()}
    labeled = sum(1 for r in records if "node_features" in r.get("meta", {}))
    size_mb = corpus_path.stat().st_size / 1e6
    print("=" * 64)
    print("SYNTHETIC CORPUS GENERATION COMPLETE")
    print("=" * 64)
    print(f"Total records      : {len(records):,} (target {TOTAL_RECORDS:,})")
    for cat, n in cat_counts.items():
        print(f"  {cat:<18}: {n:>6,}")
    print(f"Clusters           : {len(cluster_index):,} "
          f"(avg size {len(records) / max(1, len(cluster_index)):.1f})")
    print(f"Code records w/ node features (ML): {labeled:,}")
    print(f"Standalone ML nodes: {len(nodes):,}")
    print(f"Benchmark queries  : {len(queries)}")
    print(f"corpus.json size   : {size_mb:.1f} MB")
    print(f"Elapsed            : {time.time() - t0:.1f} s")
    print(f"Saved to           : {corpus_path}, {nodes_path}, {queries_path}")


if __name__ == "__main__":
    main()
