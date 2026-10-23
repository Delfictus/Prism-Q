//! Cold-start selection and admission of the actual GPU Bowyer-Watson output.
//!
//! This module never constructs or repairs tetrahedra. It selects real atoms
//! from native neighbor edges and constraint components, then qualifies the
//! immutable device readback before the physical runtime may consume it.
//! Exact dyadic integer geometry uses both represented FP32 coordinate components;
//! no epsilon, jitter, hidden hull completion or capacity truncation is used.
//! The accepted peeled domain can occupy only part of the molecular convex hull.

use crate::biprism_v10_runtime::{BiprismPreparationReadback, ConstraintGroup};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

const NV: usize = 256;
const CORE: usize = 192;
const STRIDE: usize = 1024;
const TILE_BYTES: usize = 225_408;
const TET_OFFSET: usize = 5_120;
const TET_EDGE_OFFSET: usize = 9_216;
const TET_OPPOSITE_OFFSET: usize = 21_504;
const EDGE_OFFSET: usize = 29_696;
const COUNTS_OFFSET: usize = 225_280;
const PAIRS: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
type Edge = [u16; 2];
type Face = [u16; 3];
type Tet = [u16; 4];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectionWitness {
    pub atom: u32,
    pub component: u32,
    pub depth: u32,
    pub parent_component: Option<u32>,
    pub from_atom: Option<u32>,
    pub to_atom: Option<u32>,
    pub core: bool,
}
#[derive(Clone, Debug)]
pub struct MeshSelection {
    pub selected_atom_ids: [u32; NV],
    pub core_mask: Vec<u32>,
    pub witness: Vec<SelectionWitness>,
    pub reachable_components: usize,
}

fn root(parent: &mut [usize], mut a: usize) -> usize {
    while parent[a] != a {
        parent[a] = parent[parent[a]];
        a = parent[a];
    }
    a
}
fn check_groups(n: usize, groups: &[ConstraintGroup]) -> Result<(), String> {
    for (i, g) in groups.iter().enumerate() {
        let set: BTreeSet<_> = g.atoms.iter().copied().collect();
        if set.is_empty() || set.len() != g.atoms.len() || set.iter().any(|&a| a as usize >= n) {
            return Err(format!(
                "constraint group {i} is empty, duplicated or out of bounds"
            ));
        }
    }
    Ok(())
}
fn check_closure(ids: &[u32; NV], n: usize, groups: &[ConstraintGroup]) -> Result<(), String> {
    check_groups(n, groups)?;
    let core: BTreeSet<_> = ids[..CORE].iter().copied().collect();
    for (i, g) in groups.iter().enumerate() {
        let count = g.atoms.iter().filter(|a| core.contains(a)).count();
        if count != 0 && count != g.atoms.len() {
            return Err(format!("selected core splits constraint group {i}"));
        }
    }
    Ok(())
}

/// Deterministic BFS over actual native neighbor edges, after unioning complete
/// native constraint components. A bounded subset sum selects exactly192 core
/// atoms without splitting components;64 halo atoms come from discovered shells.
/// Saturated/invalid/duplicate neighbor rows refuse rather than inventing IDs.
pub fn select_native_population(
    neighbor_list: &[i32],
    neighbor_counts: &[i32],
    atom_count: u32,
    constraints: &[ConstraintGroup],
    center: u32,
) -> Result<MeshSelection, String> {
    let n = atom_count as usize;
    if !(NV..=0x0100_0000).contains(&n)
        || center as usize >= n
        || neighbor_counts.len() != n
        || n.checked_mul(STRIDE) != Some(neighbor_list.len())
    {
        return Err("native neighbor population/allocation does not match N and N*1024".into());
    }
    check_groups(n, constraints)?;
    let mut parent: Vec<_> = (0..n).collect();
    for g in constraints {
        for &a in &g.atoms[1..] {
            let x = root(&mut parent, g.atoms[0] as usize);
            let y = root(&mut parent, a as usize);
            if x != y {
                parent[x.max(y)] = x.min(y);
            }
        }
    }
    let mut component = vec![0usize; n];
    let mut members = vec![Vec::new(); n];
    for a in 0..n {
        component[a] = root(&mut parent, a);
        members[component[a]].push(a);
        let count = neighbor_counts[a];
        if count < 0 || count as usize >= STRIDE {
            return Err(format!("native neighbor row {a} is invalid or saturated"));
        }
        let mut seen = BTreeSet::new();
        for &b in &neighbor_list[a * STRIDE..a * STRIDE + count as usize] {
            if b < 0 || b as usize >= n || b as usize == a || !seen.insert(b) {
                return Err(format!(
                    "native neighbor row {a} has invalid/self/duplicate atom {b}"
                ));
            }
        }
    }
    let start = component[center as usize];
    if members[start].len() > CORE {
        return Err("center constraint component exceeds192 atoms".into());
    }
    let mut depth = vec![u32::MAX; n];
    let mut previous = vec![None; n];
    let mut edge = vec![None; n];
    depth[start] = 0;
    let mut bfs = vec![start];
    let mut q = 0;
    while q < bfs.len() {
        let c = bfs[q];
        q += 1;
        for &a in &members[c] {
            for &bb in &neighbor_list[a * STRIDE..a * STRIDE + neighbor_counts[a] as usize] {
                let b = bb as usize;
                let d = component[b];
                if depth[d] == u32::MAX {
                    depth[d] = depth[c] + 1;
                    previous[d] = Some(c);
                    edge[d] = Some((a, b));
                    bfs.push(d);
                }
            }
        }
    }
    bfs.sort_by_key(|&c| (depth[c], c));
    let remaining = CORE - members[start].len();
    let mut choices = vec![None; remaining + 1];
    let mut reachable = vec![false; remaining + 1];
    reachable[0] = true;
    for &c in &bfs {
        if c == start {
            continue;
        }
        let size = members[c].len();
        if size > remaining {
            continue;
        }
        for sum in (size..=remaining).rev() {
            if !reachable[sum] && reachable[sum - size] {
                reachable[sum] = true;
                choices[sum] = Some((c, sum - size));
            }
        }
        if reachable[remaining] {
            break;
        }
    }
    if !reachable[remaining] {
        return Err("reachable whole constraint components cannot fill192 core atoms".into());
    }
    let mut chosen = vec![false; n];
    chosen[start] = true;
    let mut sum = remaining;
    while sum > 0 {
        let (c, p) = choices[sum].ok_or("invalid subset-sum witness")?;
        chosen[c] = true;
        sum = p;
    }
    let mut selected = vec![center];
    let mut used = vec![false; n];
    used[center as usize] = true;
    for &c in &bfs {
        if chosen[c] {
            for &a in &members[c] {
                if !used[a] {
                    selected.push(a as u32);
                    used[a] = true;
                }
            }
        }
    }
    if selected.len() != CORE {
        return Err("constraint component selection did not produce192 atoms".into());
    }
    for &c in &bfs {
        for &a in &members[c] {
            if !used[a] && selected.len() < NV {
                selected.push(a as u32);
                used[a] = true;
            }
        }
    }
    let selected_atom_ids: [u32; NV] = selected
        .try_into()
        .map_err(|_| "reachable graph has fewer than256 selected atoms")?;
    check_closure(&selected_atom_ids, n, constraints)?;
    let mut core_mask = vec![0u32; n];
    for &a in &selected_atom_ids[..CORE] {
        core_mask[a as usize] = 1;
    }
    let witness = selected_atom_ids
        .iter()
        .enumerate()
        .map(|(v, &a)| {
            let c = component[a as usize];
            SelectionWitness {
                atom: a,
                component: c as u32,
                depth: depth[c],
                parent_component: previous[c].map(|x| x as u32),
                from_atom: edge[c].map(|x| x.0 as u32),
                to_atom: edge[c].map(|x| x.1 as u32),
                core: v < CORE,
            }
        })
        .collect();
    Ok(MeshSelection {
        selected_atom_ids,
        core_mask,
        witness,
        reachable_components: bfs.len(),
    })
}

// Signed base2^32 integers. Finite FP32 coordinates on a common exact binary
// scale need at most277 bits. Degree5 in-sphere expressions need fewer than1408
// bits including sums;48 limbs provide1536 bits. No floating tolerance governs
// admission. This is fixed-domain arithmetic, not a general big-integer API.
#[derive(Clone, Copy, Debug)]
struct Exact {
    sign: i8,
    n: usize,
    a: [u32; 48],
}
impl Exact {
    fn zero() -> Self {
        Self {
            sign: 0,
            n: 0,
            a: [0; 48],
        }
    }
    fn normalize(mut self) -> Self {
        while self.n > 0 && self.a[self.n - 1] == 0 {
            self.n -= 1;
        }
        if self.n == 0 {
            self.sign = 0;
        }
        self
    }
    fn from_f32(x: f32, exponent: i32) -> Self {
        let bits = x.to_bits();
        let e = ((bits >> 23) & 255) as i32;
        let m = (bits & 0x7fffff) | if e == 0 { 0 } else { 1 << 23 };
        if m == 0 {
            return Self::zero();
        }
        let shift = (if e == 0 { -149 } else { e - 150 }) - exponent;
        debug_assert!(shift >= 0);
        let w = shift as usize / 32;
        let b = shift as usize % 32;
        let z = (m as u64) << b;
        let mut r = Self::zero();
        r.sign = if bits >> 31 == 0 { 1 } else { -1 };
        r.a[w] = z as u32;
        r.a[w + 1] = (z >> 32) as u32;
        r.n = w + 2;
        r.normalize()
    }
    fn abs_cmp(&self, b: &Self) -> std::cmp::Ordering {
        self.n.cmp(&b.n).then_with(|| {
            (0..self.n)
                .rev()
                .find_map(|i| (self.a[i] != b.a[i]).then(|| self.a[i].cmp(&b.a[i])))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }
    fn cmp(&self, b: &Self) -> std::cmp::Ordering {
        self.sign.cmp(&b.sign).then_with(|| {
            if self.sign < 0 {
                b.abs_cmp(self)
            } else {
                self.abs_cmp(b)
            }
        })
    }
    fn neg(mut self) -> Self {
        self.sign = -self.sign;
        self
    }
    fn add(self, b: Self) -> Self {
        if self.sign == 0 {
            return b;
        }
        if b.sign == 0 {
            return self;
        }
        let mut r = Self::zero();
        if self.sign == b.sign {
            r.sign = self.sign;
            r.n = self.n.max(b.n);
            let mut carry = 0u64;
            for i in 0..r.n {
                let z = self.a[i] as u64 + b.a[i] as u64 + carry;
                r.a[i] = z as u32;
                carry = z >> 32;
            }
            if carry != 0 {
                assert!(r.n < 48, "exact-domain addition bound");
                r.a[r.n] = carry as u32;
                r.n += 1;
            }
        } else {
            let (a, b) = if self.abs_cmp(&b).is_lt() {
                (b, self)
            } else {
                (self, b)
            };
            r.sign = a.sign;
            r.n = a.n;
            let mut borrow = 0i64;
            for i in 0..r.n {
                let z = a.a[i] as i64 - b.a[i] as i64 - borrow;
                r.a[i] = z as u32;
                borrow = i64::from(z < 0);
            }
            debug_assert_eq!(borrow, 0);
        }
        r.normalize()
    }
    fn sub(self, b: Self) -> Self {
        self.add(b.neg())
    }
    fn mul(self, b: Self) -> Self {
        if self.sign == 0 || b.sign == 0 {
            return Self::zero();
        }
        assert!(self.n + b.n <= 48, "exact-domain multiplication bound");
        let mut r = Self::zero();
        r.sign = self.sign * b.sign;
        r.n = self.n + b.n;
        for i in 0..self.n {
            let mut carry = 0u64;
            for j in 0..b.n {
                let z = self.a[i] as u64 * b.a[j] as u64 + r.a[i + j] as u64 + carry;
                r.a[i + j] = z as u32;
                carry = z >> 32;
            }
            r.a[i + b.n] = carry as u32;
        }
        r.normalize()
    }
}
type XVec = [Exact; 3];
fn xsub(a: XVec, b: XVec) -> XVec {
    std::array::from_fn(|i| a[i].sub(b[i]))
}
fn xcross(a: XVec, b: XVec) -> XVec {
    [
        a[1].mul(b[2]).sub(a[2].mul(b[1])),
        a[2].mul(b[0]).sub(a[0].mul(b[2])),
        a[0].mul(b[1]).sub(a[1].mul(b[0])),
    ]
}
fn xdot(a: XVec, b: XVec) -> Exact {
    a[0].mul(b[0]).add(a[1].mul(b[1])).add(a[2].mul(b[2]))
}
fn xscale(a: XVec, b: Exact) -> XVec {
    std::array::from_fn(|i| a[i].mul(b))
}
fn xadd(a: XVec, b: XVec) -> XVec {
    std::array::from_fn(|i| a[i].add(b[i]))
}
fn exact_points(points: &[[f32; 3]; NV]) -> Result<[XVec; NV], String> {
    exact_points_compensated(points,&[[0.0;3];NV])
}
fn exact_points_compensated(points:&[[f32;3];NV],low:&[[f32;3];NV])->Result<[XVec;NV],String>{
    if points.iter().chain(low).flatten().any(|x| !x.is_finite()) {
        return Err("nonfinite selected coordinate".into());
    }
    let exponent = points
        .iter().chain(low)
        .flatten()
        .filter(|x| **x != 0.0)
        .map(|x| {
            let e = ((x.to_bits() >> 23) & 255) as i32;
            if e == 0 {
                -149
            } else {
                e - 150
            }
        })
        .min()
        .ok_or("all coordinates zero")?;
    Ok(std::array::from_fn(|v| {
        std::array::from_fn(|k| Exact::from_f32(points[v][k], exponent).add(Exact::from_f32(low[v][k],exponent)))
    }))
}
fn sphere_data(p: &[XVec; NV], t: Tet) -> (Exact, XVec) {
    let a = p[t[0] as usize];
    let u = xsub(p[t[1] as usize], a);
    let v = xsub(p[t[2] as usize], a);
    let w = xsub(p[t[3] as usize], a);
    let vw = xcross(v, w);
    let wu = xcross(w, u);
    let uv = xcross(u, v);
    (
        xdot(u, vw),
        xadd(
            xadd(xscale(vw, xdot(u, u)), xscale(wu, xdot(v, v))),
            xscale(uv, xdot(w, w)),
        ),
    )
}
fn edge(a: u16, b: u16) -> Edge {
    if a < b {
        [a, b]
    } else {
        [b, a]
    }
}
fn face(t: Tet, opposite: usize) -> Face {
    let mut x = [0; 3];
    let mut k = 0;
    for (i, &v) in t.iter().enumerate() {
        if i != opposite {
            x[k] = v;
            k += 1;
        }
    }
    x.sort();
    x
}
type Graph = BTreeMap<u16, BTreeSet<u16>>;
fn connect(g: &mut Graph, a: u16, b: u16) {
    g.entry(a).or_default().insert(b);
    g.entry(b).or_default().insert(a);
}
fn connected(g: &Graph) -> bool {
    let Some(&first) = g.keys().next() else {
        return false;
    };
    let mut seen = BTreeSet::new();
    let mut queue = vec![first];
    while let Some(v) = queue.pop() {
        if seen.insert(v) {
            queue.extend(g[&v].iter().copied());
        }
    }
    seen.len() == g.len()
}
fn path_or_cycle(edges: &[Edge], boundary: bool) -> Result<(), String> {
    let mut g = Graph::new();
    let mut seen = BTreeSet::new();
    for &e in edges {
        if e[0] == e[1] || !seen.insert(e) {
            return Err("duplicate/degenerate link edge".into());
        }
        connect(&mut g, e[0], e[1]);
    }
    let ends = g.values().filter(|s| s.len() == 1).count();
    if !connected(&g)
        || g.values().any(|s| s.is_empty() || s.len() > 2)
        || ends != if boundary { 2 } else { 0 }
    {
        return Err("link is not one connected path/cycle".into());
    }
    Ok(())
}
#[derive(Clone, Copy)]
struct Surface {
    vertices: usize,
    edges: usize,
    faces: usize,
    boundary: bool,
}
fn surface(triangles: &[Face], require_closed: bool) -> Result<Surface, String> {
    let mut unique = BTreeSet::new();
    let mut edges: BTreeMap<Edge, usize> = BTreeMap::new();
    let mut stars: BTreeMap<u16, Vec<Edge>> = BTreeMap::new();
    for &t in triangles {
        if t[0] >= t[1] || t[1] >= t[2] || !unique.insert(t) {
            return Err("invalid/duplicate surface triangle".into());
        }
        for (i, j) in [(0, 1), (0, 2), (1, 2)] {
            *edges.entry(edge(t[i], t[j])).or_default() += 1;
        }
        for k in 0..3 {
            let rest: Vec<_> = t.iter().copied().filter(|&v| v != t[k]).collect();
            stars.entry(t[k]).or_default().push(edge(rest[0], rest[1]));
        }
    }
    let mut g = Graph::new();
    let mut boundary = Graph::new();
    for (&e, &n) in &edges {
        if n > 2 {
            return Err("surface edge has more than2 triangles".into());
        }
        connect(&mut g, e[0], e[1]);
        if n == 1 {
            connect(&mut boundary, e[0], e[1]);
        }
    }
    if !connected(&g) {
        return Err("surface is disconnected".into());
    }
    let is_boundary = !boundary.is_empty();
    if require_closed && is_boundary {
        return Err("boundary surface is open".into());
    }
    if is_boundary && (!connected(&boundary) || boundary.values().any(|x| x.len() != 2)) {
        return Err("surface boundary is not one circle".into());
    }
    let chi = g.len() as i64 - edges.len() as i64 + triangles.len() as i64;
    if chi != if is_boundary { 1 } else { 2 } {
        return Err("surface is not a disk/sphere by Euler characteristic".into());
    }
    for (&v, es) in &stars {
        path_or_cycle(es, boundary.contains_key(&v))?;
    }
    Ok(Surface {
        vertices: g.len(),
        edges: edges.len(),
        faces: triangles.len(),
        boundary: is_boundary,
    })
}

#[derive(Clone, Debug)]
pub struct MeshAdmission {
    pub vertices: usize,
    pub tetrahedra: usize,
    pub edges: usize,
    pub boundary_faces: usize,
    pub boundary_vertices: usize,
    pub exact_empty_sphere_tests: usize,
    pub exact_sat_pairs: usize,
    pub volume_au_cubed: f64,
    /// This accepts the exported subdomain; it does not certify full hull coverage.
    pub full_convex_hull_coverage_claimed: bool,
}
fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}
fn u16_at(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes(b[o..o + 2].try_into().unwrap())
}
struct Geometry {
    tet: Tet,
    edges: [XVec; 6],
    normals: [XVec; 4],
    lo: XVec,
    hi: XVec,
}
fn separates(axis: XVec, a: Tet, b: Tet, p: &[XVec; NV]) -> bool {
    if axis.iter().all(|x| x.sign == 0) {
        return false;
    }
    let origin = p[a[0] as usize];
    let pa = a.map(|v| xdot(axis, xsub(p[v as usize], origin)));
    let pb = b.map(|v| xdot(axis, xsub(p[v as usize], origin)));
    let min_a = pa.iter().min_by(|x, y| x.cmp(y)).unwrap();
    let max_a = pa.iter().max_by(|x, y| x.cmp(y)).unwrap();
    let min_b = pb.iter().min_by(|x, y| x.cmp(y)).unwrap();
    let max_b = pb.iter().max_by(|x, y| x.cmp(y)).unwrap();
    !max_a.cmp(min_b).is_gt() || !max_b.cmp(min_a).is_gt()
}
fn disjoint_interiors(a: &Geometry, b: &Geometry, p: &[XVec; NV]) -> bool {
    for axis in a.normals.iter().chain(&b.normals) {
        if separates(*axis, a.tet, b.tet, p) {
            return true;
        }
    }
    for ea in &a.edges {
        for eb in &b.edges {
            if separates(xcross(*ea, *eb), a.tet, b.tet, p) {
                return true;
            }
        }
    }
    false
}

/// Strong admission of one immutable GPU tile. The byte offsets are the actual
///225408B CUDA tile ABI, independently compiler-checked in the source receipts.
/// Coordinates are exactly the seeder's captured FP32 atomic-unit positions.
pub fn validate_tile_geometry(
    bytes: &[u8],
    points: &[[f32; 3]; NV],
    atom_count: u32,
    constraints: &[ConstraintGroup],
    target_tets: u32,
) -> Result<MeshAdmission, String> {
    validate_tile_geometry_compensated(bytes,points,&[[0.0;3];NV],atom_count,constraints,target_tets)
}

pub fn validate_tile_geometry_compensated(bytes:&[u8],points:&[[f32;3];NV],low:&[[f32;3];NV],
    atom_count:u32,constraints:&[ConstraintGroup],target_tets:u32)->Result<MeshAdmission,String>{
    if bytes.len() != TILE_BYTES
        || target_tets == 0
        || target_tets > 1024
        || atom_count < 256
        || atom_count > 0x0100_0000
    {
        return Err("tile/input shape outside supported ABI".into());
    }
    let nt = u32_at(bytes, COUNTS_OFFSET) as usize;
    let ne = u32_at(bytes, COUNTS_OFFSET + 4) as usize;
    if nt == 0 || nt > target_tets as usize || nt > 1024 || ne == 0 || ne > 1536 {
        return Err("active tile capacity/count refusal".into());
    }
    let ids: [u32; NV] = std::array::from_fn(|v| u32_at(bytes, v * 4) & 0x00ff_ffff);
    if ids.iter().any(|&a| a >= atom_count)
        || ids.iter().copied().collect::<BTreeSet<_>>().len() != NV
    {
        return Err("tile atom identity duplicated or out of bounds".into());
    }
    check_closure(&ids, atom_count as usize, constraints)?;
    for v in 0..NV {
        for k in 0..3 {
            if u32_at(bytes, 1024 + v * 16 + k * 4) != points[v][k].to_bits() {
                return Err(format!(
                    "tile reference coordinate differs from captured input at{v}/{k}"
                ));
            }
        }
    }
    let p = exact_points_compensated(points,low)?;
    let mut edge_ids = BTreeMap::new();
    for e in 0..ne {
        let ep = [
            bytes[EDGE_OFFSET + 2 * e] as u16,
            bytes[EDGE_OFFSET + 2 * e + 1] as u16,
        ];
        if ep[0] >= ep[1] || edge_ids.insert(ep, e).is_some() {
            return Err(format!("invalid or duplicate edge{e}"));
        }
    }
    let mut tets = Vec::with_capacity(nt);
    let mut unique = BTreeSet::new();
    let mut faces: BTreeMap<Face, Vec<(usize, usize)>> = BTreeMap::new();
    let mut links: BTreeMap<Edge, Vec<Edge>> = BTreeMap::new();
    let mut stars: Vec<Vec<Face>> = vec![Vec::new(); NV];
    let mut geometry = Vec::with_capacity(nt);
    let mut sphere_tests = 0usize;
    let mut volume = 0.0;
    for ti in 0..nt {
        let t: Tet = std::array::from_fn(|k| bytes[TET_OFFSET + ti * 4 + k] as u16);
        let mut key = t;
        key.sort();
        if key.windows(2).any(|w| w[0] == w[1]) || !unique.insert(key) {
            return Err(format!("degenerate or duplicate tetrahedron{ti}"));
        }
        let (det, numer) = sphere_data(&p, t);
        if det.sign <= 0 {
            return Err(format!("nonpositive exact orientation at tetrahedron{ti}"));
        }
        for v in 0..NV {
            if t.contains(&(v as u16)) {
                continue;
            }
            let q = xsub(p[v], p[t[0] as usize]);
            let test = det.mul(xdot(q, q)).sub(xdot(numer, q));
            sphere_tests += 1;
            if test.sign <= 0 {
                return Err(format!(
                    "nonempty or cospherical exact circumsphere at tetrahedron{ti}, vertex{v}"
                ));
            }
        }
        for (k, &(a, b)) in PAIRS.iter().enumerate() {
            let ep = edge(t[a], t[b]);
            let eid = u16_at(bytes, TET_EDGE_OFFSET + ti * 12 + k * 2) as usize;
            if eid >= ne || edge_ids.get(&ep) != Some(&eid) {
                return Err(format!("tetrahedron{ti} edge-slot{k} incidence mismatch"));
            }
            let op: Vec<_> = t
                .iter()
                .copied()
                .filter(|&v| v != t[a] && v != t[b])
                .collect();
            links.entry(ep).or_default().push(edge(op[0], op[1]));
        }
        for k in 0..4 {
            let f = face(t, k);
            faces.entry(f).or_default().push((ti, k));
            stars[t[k] as usize].push(f);
        }
        let edges = std::array::from_fn(|k| {
            let (a, b) = PAIRS[k];
            xsub(p[t[b] as usize], p[t[a] as usize])
        });
        let normals = std::array::from_fn(|k| {
            let f = face(t, k);
            xcross(
                xsub(p[f[1] as usize], p[f[0] as usize]),
                xsub(p[f[2] as usize], p[f[0] as usize]),
            )
        });
        let lo=std::array::from_fn(|k|t.iter().map(|&v|p[v as usize][k]).min_by(|a,b|a.cmp(b)).unwrap());
        let hi=std::array::from_fn(|k|t.iter().map(|&v|p[v as usize][k]).max_by(|a,b|a.cmp(b)).unwrap());
        let a=t[0] as usize;
        let delta=|b:usize,k:usize|(f64::from(points[b][k])-f64::from(points[a][k]))+(f64::from(low[b][k])-f64::from(low[a][k]));
        let u=std::array::from_fn::<_,3,_>(|k|delta(t[1] as usize,k));
        let v=std::array::from_fn::<_,3,_>(|k|delta(t[2] as usize,k));
        let w=std::array::from_fn::<_,3,_>(|k|delta(t[3] as usize,k));
        volume += (u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0])
            + u[2] * (v[0] * w[1] - v[1] * w[0]))
            / 6.0;
        geometry.push(Geometry {
            tet: t,
            edges,
            normals,
            lo,
            hi,
        });
        tets.push(t);
    }
    if links.len() != ne || stars.iter().any(Vec::is_empty) {
        return Err("unused active edge or missing selected vertex".into());
    }
    let mut boundary = Vec::new();
    let mut boundary_edges = BTreeSet::new();
    let mut dual = vec![Vec::new(); nt];
    for (f, owners) in &faces {
        if owners.is_empty() || owners.len() > 2 {
            return Err("face has invalid multiplicity".into());
        }
        if owners.len() == 1 {
            boundary.push(*f);
            for (a, b) in [(0, 1), (0, 2), (1, 2)] {
                boundary_edges.insert(edge(f[a], f[b]));
            }
        }
        for (i, &(t, k)) in owners.iter().enumerate() {
            let actual = u16_at(bytes, TET_OPPOSITE_OFFSET + t * 8 + k * 2);
            let expected = if owners.len() == 1 {
                u16::MAX
            } else {
                owners[1 - i].0 as u16
            };
            if actual != expected {
                return Err(format!(
                    "tetrahedron{t} opposite-face{k} adjacency mismatch"
                ));
            }
            if expected != u16::MAX {
                dual[t].push(expected as usize);
            }
        }
    }
    let mut visited = vec![false; nt];
    let mut queue = VecDeque::from([0usize]);
    let mut n = 0;
    while let Some(t) = queue.pop_front() {
        if !visited[t] {
            visited[t] = true;
            n += 1;
            queue.extend(dual[t].iter().copied());
        }
    }
    if n != nt {
        return Err("tetrahedral dual graph is disconnected".into());
    }
    for (e, es) in &links {
        path_or_cycle(es, boundary_edges.contains(e)).map_err(|why| format!("edge{e:?}: {why}"))?;
    }
    for (v, fs) in stars.iter().enumerate() {
        surface(fs, false).map_err(|why| format!("vertex{v} link: {why}"))?;
    }
    let outer = surface(&boundary, true).map_err(|why| format!("boundary: {why}"))?;
    if NV as i64 - ne as i64 + faces.len() as i64 - nt as i64 != 1 {
        return Err("tetrahedral complex Euler characteristic is not1".into());
    }
    let mut sat_pairs = 0;
    for i in 0..nt {
        for j in i + 1..nt {
            let a = &geometry[i];
            let b = &geometry[j];
            if (0..3).any(|k| !a.hi[k].cmp(&b.lo[k]).is_gt() || !b.hi[k].cmp(&a.lo[k]).is_gt()) {
                continue;
            }
            sat_pairs += 1;
            if !disjoint_interiors(a, b, &p) {
                return Err(format!("positive-volume tetrahedron overlap{i}/{j}"));
            }
        }
    }
    if !volume.is_finite() || volume <= 0.0 {
        return Err("nonfinite/empty represented mesh volume".into());
    }
    debug_assert!(!outer.boundary && outer.vertices + outer.faces == outer.edges + 2);
    Ok(MeshAdmission {
        vertices: NV,
        tetrahedra: nt,
        edges: ne,
        boundary_faces: boundary.len(),
        boundary_vertices: outer.vertices,
        exact_empty_sphere_tests: sphere_tests,
        exact_sat_pairs: sat_pairs,
        volume_au_cubed: volume,
        full_convex_hull_coverage_claimed: false,
    })
}

/// Callback for `BiprismV10DeviceState::accept_preparation`. The runtime first
/// binds generation/allocation and all8 producer statuses; this adds independent
/// topology and exact geometric admission of every immutable readback leaf.
pub fn validate_mesh_readback(
    readback: &BiprismPreparationReadback,
    atom_count: u32,
    constraints: &[ConstraintGroup],
) -> Result<[MeshAdmission; 8], String> {
    if readback.tile_bytes().len() != 8 * TILE_BYTES {
        return Err("eight-leaf tile readback size mismatch".into());
    }
    let mut reports = Vec::with_capacity(8);
    for leaf in 0..8 {
        reports.push(
            validate_tile_geometry_compensated(
                &readback.tile_bytes()[leaf * TILE_BYTES..(leaf + 1) * TILE_BYTES],
                &readback.selected_positions_au()[leaf],
                &readback.selected_positions_au_low()[leaf],
                atom_count,
                constraints,
                readback.target_tets(),
            )
            .map_err(|why| format!("leaf{leaf}: {why}"))?,
        );
    }
    reports
        .try_into()
        .map_err(|_| "eight-leaf report cardinality mismatch".into())
}

#[cfg(test)]
mod tests {
    #[test]
    fn compensated_predicates_keep_low_only_orientation() {
        let h=[[64.0f32;3];NV];let mut l=[[0.0f32;3];NV];
        l[1][0]=1e-7;l[2][1]=1e-7;l[3][2]=1e-7;
        let p=exact_points_compensated(&h,&l).unwrap();
        assert!(sphere_data(&p,[0,1,2,3]).0.sign>0);
        assert_eq!(sphere_data(&exact_points(&h).unwrap(),[0,1,2,3]).0.sign,0);
        l[3][2]=-1e-7;
        assert!(sphere_data(&exact_points_compensated(&h,&l).unwrap(),[0,1,2,3]).0.sign<0);
    }

    use super::*;
    #[test]
    fn exact_sphere_and_orientation_controls() {
        for scale in [1.0f32, 1.0e-30, 1.0e30] {
            let mut p = [[0.0f32; 3]; NV];
            p[1] = [scale, 0.0, 0.0];
            p[2] = [0.0, scale, 0.0];
            p[3] = [0.0, 0.0, scale];
            p[4] = [scale * 0.25; 3];
            p[5] = [scale * 2.0; 3];
            p[6] = [scale; 3];
            let q = exact_points(&p).unwrap();
            let (d, n) = sphere_data(&q, [0, 1, 2, 3]);
            assert_eq!(d.sign, 1);
            for (v, want) in [(4, -1), (5, 1), (6, 0)] {
                let x = xsub(q[v], q[0]);
                assert_eq!(d.mul(xdot(x, x)).sub(xdot(n, x)).sign, want);
            }
            assert_eq!(sphere_data(&q, [0, 2, 1, 3]).0.sign, -1);
        }
    }
    #[test]
    fn full_fp32_exponent_domain_is_exact_and_bounded() {
        let mut p = [[0.0f32; 3]; NV];
        p[0] = [f32::from_bits(1), 0.0, 0.0];
        p[1] = [f32::MAX, 0.0, 0.0];
        p[2] = [0.0, f32::MAX, 0.0];
        p[3] = [0.0, 0.0, f32::MAX];
        p[4] = [-f32::MAX; 3];
        let q = exact_points(&p).unwrap();
        let (d, n) = sphere_data(&q, [0, 1, 2, 3]);
        assert_eq!(d.sign, 1);
        let x = xsub(q[4], q[0]);
        assert_eq!(d.mul(xdot(x, x)).sub(xdot(n, x)).sign, 1);
    }
    #[test]
    fn surface_rejects_annulus_and_disconnected_link() {
        assert!(path_or_cycle(&[[0, 1], [1, 2], [3, 4], [4, 5]], true).is_err());
        assert!(surface(&[[0, 1, 2], [0, 2, 3]], false).is_ok());
        assert!(surface(&[[0, 1, 2], [0, 2, 3]], true).is_err());
        assert!(surface(&[[0, 1, 2], [0, 1, 2]], false).is_err());
        let annulus = [
            [0, 1, 5],
            [0, 4, 5],
            [1, 2, 6],
            [1, 5, 6],
            [2, 3, 7],
            [2, 6, 7],
            [0, 3, 4],
            [3, 4, 7],
        ];
        assert!(surface(&annulus, false).is_err());
    }
    #[test]
    fn exact_sat_distinguishes_overlap_contact_and_separation() {
        for (shift, separated) in [(0.1f32, false), (1.0, true), (2.0, true)] {
            let mut points = [[0.0f32; 3]; NV];
            points[1] = [1.0, 0.0, 0.0];
            points[2] = [0.0, 1.0, 0.0];
            points[3] = [0.0, 0.0, 1.0];
            for i in 0..4 {
                points[i + 4] = points[i];
                points[i + 4][0] += shift;
            }
            let p = exact_points(&points).unwrap();
            let make = |tet: Tet| Geometry {
                tet,
                edges: std::array::from_fn(|i| {
                    let (a, b) = PAIRS[i];
                    xsub(p[tet[b] as usize], p[tet[a] as usize])
                }),
                normals: std::array::from_fn(|i| {
                    let f = face(tet, i);
                    xcross(
                        xsub(p[f[1] as usize], p[f[0] as usize]),
                        xsub(p[f[2] as usize], p[f[0] as usize]),
                    )
                }),
                lo: [Exact::zero(); 3],
                hi: [Exact::zero(); 3],
            };
            assert_eq!(
                disjoint_interiors(&make([0, 1, 2, 3]), &make([4, 5, 6, 7]), &p),
                separated
            );
        }
    }
    fn chain(n: usize) -> (Vec<i32>, Vec<i32>) {
        let mut list = vec![-1; n * STRIDE];
        let mut count = vec![0; n];
        for a in 0..n {
            for b in [a.checked_sub(1), (a + 1 < n).then_some(a + 1)]
                .into_iter()
                .flatten()
            {
                list[a * STRIDE + count[a] as usize] = b as i32;
                count[a] += 1;
            }
        }
        (list, count)
    }
    #[test]
    fn real_graph_constraint_components_and_witness() {
        let (list, count) = chain(260);
        let groups = (0..260)
            .step_by(2)
            .map(|a| ConstraintGroup {
                atoms: vec![a, a + 1],
            })
            .collect::<Vec<_>>();
        let s = select_native_population(&list, &count, 260, &groups, 0).unwrap();
        assert_eq!(
            s.selected_atom_ids,
            <[u32; 256]>::try_from((0..256).collect::<Vec<u32>>()).unwrap()
        );
        assert_eq!(s.core_mask.iter().sum::<u32>(), 192);
        for row in &s.witness {
            if row.depth > 0 {
                let a = row.from_atom.unwrap() as usize;
                let b = row.to_atom.unwrap() as i32;
                assert!(list[a * STRIDE..a * STRIDE + count[a] as usize].contains(&b));
            }
        }
    }
    #[test]
    fn selection_refuses_padding_and_invalid_native_rows() {
        let (mut list, mut count) = chain(260);
        count[191] = 1;
        list[191 * STRIDE] = 190;
        assert!(select_native_population(&list, &count, 260, &[], 0).is_err());
        let (list, mut count) = chain(260);
        count[7] = 1024;
        assert!(select_native_population(&list, &count, 260, &[], 0).is_err());
        let (list, count) = chain(260);
        assert!(select_native_population(
            &list,
            &count,
            260,
            &[ConstraintGroup {
                atoms: (0..193).collect()
            }],
            0
        )
        .is_err());
    }
}
