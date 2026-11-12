// ---------------------------------------------------------------------------
// lbm3d-perm: permeability tensor of a 3D triply periodic porous cell
//
// D3Q19 lattice Boltzmann, BGK collision (tau = 1), Guo (2002) body forcing,
// halfway bounce-back on solid links, periodic in all three axes, double
// precision distributions.
//
// Darcy's law for a periodic cell is q_i = (K_ij / mu) F_j, so the full 3x3
// tensor needs three solves of the same structure, one per forcing axis, each
// contributing one column of K:
//
//     run x:  k_xx = mu q_x / F,  k_yx = mu q_y / F,  k_zx = mu q_z / F
//     run y:  k_xy, k_yy, k_zy
//     run z:  k_xz, k_yz, k_zz
//
// q_i = mean(u_i) over the whole cell (pore and solid), the Darcy superficial
// velocity, with u the physical Guo velocity (sum f e + F/2)/rho; mu = cs^2
// (tau - 1/2). A periodic cell has no inlet/outlet boundary layers, so the
// tensor is that of the cell itself. All three solves run in one invocation:
// the structure is read once, and since K = K^T is exact for Stokes flow, the
// three reciprocity residuals |K_ij - K_ji| / tr K are per-sample error bars on
// the off-diagonals. The CSV reports K, the symmetrised off-diagonals
// k_ij = (K_ij + K_ji)/2, the eigenvalues k1 >= k2 >= k3 and eigenvectors of the
// symmetrised tensor, and its fractional anisotropy.
//
// Convergence: every --lag steps, compare each component of the domain-averaged
// flux with its value one lag earlier, and stop once all three have moved by
// less than eps * ||q|| on --hold consecutive checks. The tolerance is relative
// to the magnitude of the whole flux vector, not to the component tested: the
// transverse components (the off-diagonals) are small by construction and
// exactly zero for a blocked direction, so a per-component relative test could
// never be satisfied. In double precision BGK reaches a genuine fixed point, so
// no statistical tolerance is needed; a short averaging phase after convergence
// still reports the standard error se_* of the flux as a check. A direction that
// hits --max-steps is flagged conv_* = 0 and the exit code is 3.
//
// Structure input is the .raw format written by scripts/structure_io3d.py:
// three little-endian int32 (nx, ny, nz) then nx*ny*nz uint8, 1 = solid, with
// x varying fastest. No external libraries are needed.
//
//   Usage: lbm3d-perm <structure.raw> <results.csv> [options]
//          (run without arguments for the option list)
//
// One row is appended to <results.csv> (with a header if the file is empty);
// exit code 0 = all three directions converged, 3 = at least one did not,
// 2 = error.
// ---------------------------------------------------------------------------

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

// --- D3Q19 -----------------------------------------------------------------
static const int Q = 19;
static const int ex[Q] = {0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1, 1,-1, 0, 0, 0, 0};
static const int ey[Q] = {0, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 0, 0, 1,-1, 1,-1};
static const int ez[Q] = {0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1,-1,-1, 1, 1,-1,-1, 1};
static const double wq[Q] = {
    1.0/3.0,
    1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0};
static int opp[Q];

static void build_opp()
{
    for(int k = 0; k < Q; k++)
        for(int m = 0; m < Q; m++)
            if(ex[m] == -ex[k] && ey[m] == -ey[k] && ez[m] == -ez[k]) { opp[k] = m; break; }
}

static const double TAU     = 1.0;
static const double CS2     = 1.0/3.0;
static const double MU      = CS2 * (TAU - 0.5);   // kinematic viscosity, rho0 = 1
static const double FX_BASE = 2.5e-07;             // body force at force factor 1

// --- configuration ---------------------------------------------------------
struct Config
{
    double ff        = 1.0;
    double eps       = 1e-5;   // drift tolerance, relative to ||q||
    long   lag       = 2000;
    int    hold      = 2;
    long   min_steps = 4000;
    long   max_steps = 200000;
    long   avg_steps = 2000;
    string velocity_prefix;
    bool   quiet      = false;
    bool   validate   = false;
};

struct Solve
{
    double q[3]  = {0,0,0};
    double se[3] = {0,0,0};
    long   steps = 0;
    int    converged = 0;
};

static void die(const string &msg)
{
    cerr << "lbm3d-perm: " << msg << endl;
    exit(2);
}

// --- lattice state ---------------------------------------------------------
static int  NX, NY, NZ;
static long NN;
static vector<uint8_t> solid;
static vector<double>  f0, f1;                 // NN*Q
static vector<double>  U, V, W, RHO;           // NN
static double Fx, Fy, Fz;
static long   n_pore;
static bool   primed = false;                  // f0 holds post-collision values

static inline long idx(int x, int y, int z) { return (long(z)*NY + y)*NX + x; }

// Precomputed wrapped neighbour coordinates for the *pull* direction (-e_k).
// Tiny (Q * N per axis) so they stay in cache, and they keep the modulo out of
// the inner loop.
static vector<int> pmx, pmy, pmz;

static void build_neighbour_tables()
{
    pmx.resize(size_t(Q)*NX); pmy.resize(size_t(Q)*NY); pmz.resize(size_t(Q)*NZ);
    for(int k = 0; k < Q; k++)
    {
        for(int x = 0; x < NX; x++) pmx[size_t(k)*NX + x] = ((x - ex[k]) % NX + NX) % NX;
        for(int y = 0; y < NY; y++) pmy[size_t(k)*NY + y] = ((y - ey[k]) % NY + NY) % NY;
        for(int z = 0; z < NZ; z++) pmz[size_t(k)*NZ + z] = ((z - ez[k]) % NZ + NZ) % NZ;
    }
}

static void read_structure(const string &path)
{
    ifstream in(path.c_str(), ios::binary);
    if(!in) die("cannot open " + path);
    int32_t hdr[3];
    in.read(reinterpret_cast<char*>(hdr), 12);
    if(!in) die("cannot read header of " + path);
    NX = hdr[0]; NY = hdr[1]; NZ = hdr[2];
    if(NX <= 0 || NY <= 0 || NZ <= 0) die("bad dimensions in " + path);
    NN = long(NX) * NY * NZ;
    solid.resize(NN);
    in.read(reinterpret_cast<char*>(solid.data()), NN);
    if(!in) die("truncated payload in " + path);
    in.peek();
    if(!in.eof()) die("trailing bytes in " + path);

    n_pore = 0;
    for(long i = 0; i < NN; i++) { if(solid[i] > 1) die("non-binary voxel"); if(!solid[i]) n_pore++; }
    if(n_pore == 0) die("structure has no pore space");
}

static void allocate()
{
    f0.assign(NN*Q, 0.0); f1.assign(NN*Q, 0.0);
    U.assign(NN, 0.0); V.assign(NN, 0.0); W.assign(NN, 0.0); RHO.assign(NN, 1.0);
}

// Reset to a uniform quiescent state, so each forcing direction starts from
// the same place the first one did.  `solid` is deliberately left alone.
static void reset_distributions()
{
    for(long i = 0; i < NN; i++)
        for(int k = 0; k < Q; k++) f0[i*Q+k] = f1[i*Q+k] = wq[k];
    fill(U.begin(), U.end(), 0.0);
    fill(V.begin(), V.end(), 0.0);
    fill(W.begin(), W.end(), 0.0);
    fill(RHO.begin(), RHO.end(), 1.0);
    primed = false;
}

// Macroscopic moments + BGK collision with Guo forcing for one fluid node:
// reads the pre-collision distributions g, writes the post-collision ones to out
// (g == out is allowed).  U,V,W are never written at solid nodes, so they stay
// exactly 0 there.
static inline void collide_node(long i, const double *g, double *out)
{
    double r = 0, ux = 0, uy = 0, uz = 0;
    for(int k = 0; k < Q; k++)
    {
        r += g[k];
        ux += g[k]*ex[k]; uy += g[k]*ey[k]; uz += g[k]*ez[k];
    }
    // Guo forcing: the physical velocity carries half the body force.
    RHO[i] = r;
    ux = U[i] = (ux + 0.5*Fx)/r;
    uy = V[i] = (uy + 0.5*Fy)/r;
    uz = W[i] = (uz + 0.5*Fz)/r;

    const double u2 = ux*ux + uy*uy + uz*uz;
    const double pref = 1.0 - 0.5/TAU;
    for(int k = 0; k < Q; k++)
    {
        const double eu = ex[k]*ux + ey[k]*uy + ez[k]*uz;
        const double feq = wq[k]*r*(1.0 + 3.0*eu + 4.5*eu*eu - 1.5*u2);
        // Guo et al. (2002) source term, with cs^2 = 1/3:
        //   S_k = w_k (1 - 1/2tau) [ 3(e_k - u) + 9(e_k.u) e_k ] . F
        const double sk = wq[k] * pref *
            ( 3.0*((ex[k]-ux)*Fx + (ey[k]-uy)*Fy + (ez[k]-uz)*Fz)
            + 9.0*eu*(ex[k]*Fx + ey[k]*Fy + ez[k]*Fz) );
        out[k] = g[k] + ((feq - g[k]) / TAU + sk);
    }
}

// Collision alone, in place on f0.  Used once per solve, to turn the initial
// (pre-collision) state into the post-collision state the fused step expects.
static void collide_only()
{
    for(long i = 0; i < NN; i++)
        if(!solid[i]) collide_node(i, &f0[i*Q], &f0[i*Q]);
}

// One fused step: stream (pull) then collide, f0 -> f1, fluid nodes only.
//
// Halfway bounce-back on links, periodic in all three axes, as a *pull*:
//
//     p = i - e_k ;  g[k] = solid[p] ? f0[i][opp[k]] : f0[p][k]
//
// then g is collided straight into f1[i].  f0 holds POST-collision values, so
// one step reads the distribution array once and writes it once.  Solid nodes
// are skipped entirely: their distributions are never read (a pull from a solid
// p takes the bounce-back branch instead), so they need neither collision nor
// zeroing.
//
// This is equivalent to a separate collide + stream per step: lbm_step() primes
// with collide_only() as its first step, after which every step computes
// U,V,W from the same pre-collision state.
static void stream_collide()
{
    double g[Q];
    for(int z = 0; z < NZ; z++)
    for(int y = 0; y < NY; y++)
    for(int x = 0; x < NX; x++)
    {
        const long i = idx(x, y, z);
        if(solid[i]) continue;
        for(int k = 0; k < Q; k++)
        {
            const long p = idx(pmx[size_t(k)*NX + x],
                               pmy[size_t(k)*NY + y],
                               pmz[size_t(k)*NZ + z]);
            g[k] = solid[p] ? f0[i*Q + opp[k]] : f0[p*Q + k];
        }
        collide_node(i, g, &f1[i*Q]);
    }
    f0.swap(f1);
}

static void lbm_step()
{
    if(!primed) { collide_only(); primed = true; }
    else          stream_collide();
}

// q_i = mean(u_i) over the whole cell: the Darcy superficial velocity.
static void volumeavg(double q[3])
{
    double su = 0, sv = 0, sw = 0;
    for(long i = 0; i < NN; i++) { su += U[i]; sv += V[i]; sw += W[i]; }
    q[0] = su/double(NN); q[1] = sv/double(NN); q[2] = sw/double(NN);
}

static void export_velocity(const string &path)
{
    ofstream out(path.c_str(), ios::binary);
    if(!out) die("cannot write " + path);
    int32_t hdr[4] = {NX, NY, NZ, 3};
    out.write(reinterpret_cast<const char*>(hdr), 16);
    vector<float> buf(3);
    for(long i = 0; i < NN; i++)                 // x fastest, as in the .raw
    {
        buf[0] = float(U[i]); buf[1] = float(V[i]); buf[2] = float(W[i]);
        out.write(reinterpret_cast<const char*>(buf.data()), 12);
    }
}

// --- one forcing direction -------------------------------------------------
static Solve solve(double fx, double fy, double fz, const Config &cfg, const char *tag)
{
    Solve out;
    reset_distributions();
    Fx = fx; Fy = fy; Fz = fz;

    long   s = 0;
    int    hold = 0;
    bool   have_prev = false;
    double qprev[3] = {0, 0, 0};

    // --- phase 1: run until the flux stops changing ------------------------
    while(s < cfg.max_steps)
    {
        lbm_step(); s++;
        if(s % cfg.lag) continue;

        double q[3];
        volumeavg(q);
        for(int c = 0; c < 3; c++)
            if(!std::isfinite(q[c]))
                die(string("non-finite flux in run ") + tag + " at step " + to_string(s));

        const double norm = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2]);
        double drift = 0.0;
        for(int c = 0; c < 3; c++) drift = max(drift, fabs(q[c] - qprev[c]));

        if(have_prev && s >= cfg.min_steps && drift < cfg.eps * norm) hold++;
        else                                                          hold = 0;

        if(!cfg.quiet)
            cout << "# " << tag << " " << s << "  q=(" << q[0] << ", " << q[1]
                 << ", " << q[2] << ")  drift/tol="
                 << (norm > 0 ? drift / (cfg.eps * norm) : 0.0)
                 << "  hold=" << hold << endl;

        for(int c = 0; c < 3; c++) qprev[c] = q[c];
        have_prev = true;

        if(hold >= cfg.hold) break;
    }

    out.converged = (hold >= cfg.hold) ? 1 : 0;
    if(!out.converged && !cfg.quiet)
        cerr << "# " << tag << ": max-steps reached without convergence" << endl;

    // --- phase 2: average, and report the spread ---------------------------
    // Not needed for accuracy (in double there is no fluctuation left to
    // average), but se_* records per sample that this is so.
    double sum[3] = {0,0,0}, sumsq[3] = {0,0,0};
    long   n = 0;
    for(long i = 0; i < cfg.avg_steps; i++)
    {
        lbm_step(); s++;
        double q[3]; volumeavg(q);
        for(int c = 0; c < 3; c++) { sum[c] += q[c]; sumsq[c] += q[c]*q[c]; }
        n++;
    }

    if(n == 0) volumeavg(out.q);
    else
        for(int c = 0; c < 3; c++)
        {
            out.q[c] = sum[c] / n;
            if(n > 1)
            {
                const double var = (sumsq[c] - n*out.q[c]*out.q[c]) / (n - 1);
                out.se[c] = sqrt(max(var, 0.0) / n);
            }
        }

    out.steps = s;

    if(!cfg.velocity_prefix.empty())
        export_velocity(cfg.velocity_prefix + "." + tag + ".vel");

    return out;
}

// --- symmetric 3x3 eigen-decomposition (cyclic Jacobi) ---------------------
static void jacobi3(const double Ain[3][3], double val[3], double vec[3][3])
{
    double a[3][3];
    memcpy(a, Ain, sizeof(a));
    for(int i = 0; i < 3; i++) for(int j = 0; j < 3; j++) vec[i][j] = (i == j) ? 1.0 : 0.0;

    for(int sweep = 0; sweep < 50; sweep++)
    {
        double off = fabs(a[0][1]) + fabs(a[0][2]) + fabs(a[1][2]);
        if(off < 1e-18) break;
        for(int p = 0; p < 2; p++)
        for(int q = p+1; q < 3; q++)
        {
            if(fabs(a[p][q]) < 1e-300) continue;
            const double theta = (a[q][q] - a[p][p]) / (2.0*a[p][q]);
            const double t = (theta >= 0 ? 1.0 : -1.0) /
                             (fabs(theta) + sqrt(theta*theta + 1.0));
            const double c = 1.0/sqrt(t*t + 1.0), s = t*c;
            for(int k = 0; k < 3; k++)
            {
                const double akp = a[k][p], akq = a[k][q];
                a[k][p] = c*akp - s*akq;  a[k][q] = s*akp + c*akq;
            }
            for(int k = 0; k < 3; k++)
            {
                const double apk = a[p][k], aqk = a[q][k];
                a[p][k] = c*apk - s*aqk;  a[q][k] = s*apk + c*aqk;
                const double vkp = vec[k][p], vkq = vec[k][q];
                vec[k][p] = c*vkp - s*vkq; vec[k][q] = s*vkp + c*vkq;
            }
        }
    }
    for(int i = 0; i < 3; i++) val[i] = a[i][i];

    // sort descending, carrying the eigenvectors
    for(int i = 0; i < 2; i++)
    for(int j = i+1; j < 3; j++)
        if(val[j] > val[i])
        {
            swap(val[i], val[j]);
            for(int k = 0; k < 3; k++) swap(vec[k][i], vec[k][j]);
        }
}

// ---------------------------------------------------------------------------
static void usage()
{
    cerr <<
"Usage: lbm3d-perm <structure.raw> <results.csv> [options]\n"
"\n"
"Solves a triply periodic cell three times (force along +x, +y, +z) and\n"
"appends one row with the full 3x3 permeability tensor to <results.csv>.\n"
"\n"
"  --ff F            force factor; body force = 2.5e-07 * F   (default 1)\n"
"  --eps E           drift tolerance, relative to |q|         (default 1e-5)\n"
"  --lag N           steps between convergence checks         (default 2000)\n"
"  --hold K          consecutive passing checks required      (default 2)\n"
"  --min-steps N     no convergence before this step          (default 4000)\n"
"  --max-steps N     give up after this many steps            (default 200000)\n"
"  --avg-steps N     averaging phase after convergence        (default 2000)\n"
"  --velocity PREFIX write PREFIX.fx.vel, PREFIX.fy.vel, PREFIX.fz.vel\n"
"  --validate        extra solve along (1,1,1)/sqrt(3); reports the flux\n"
"                    predicted by K against the measured one (Koza09 test)\n"
"  --quiet           suppress per-check progress\n";
}

static long need_long(int argc, char **argv, int &i)
{
    if(i + 1 >= argc) { usage(); die(string("missing value for ") + argv[i]); }
    return strtol(argv[++i], nullptr, 10);
}
static double need_double(int argc, char **argv, int &i)
{
    if(i + 1 >= argc) { usage(); die(string("missing value for ") + argv[i]); }
    return strtod(argv[++i], nullptr);
}

int main(int argc, char **argv)
{
    if(argc < 3) { usage(); return 2; }
    build_opp();

    const string structure  = argv[1];
    const string resultfile = argv[2];
    Config cfg;

    for(int i = 3; i < argc; i++)
    {
        const string a = argv[i];
        if     (a == "--ff")         cfg.ff         = need_double(argc, argv, i);
        else if(a == "--eps")        cfg.eps        = need_double(argc, argv, i);
        else if(a == "--lag")        cfg.lag        = need_long(argc, argv, i);
        else if(a == "--hold")       cfg.hold       = int(need_long(argc, argv, i));
        else if(a == "--min-steps")  cfg.min_steps  = need_long(argc, argv, i);
        else if(a == "--max-steps")  cfg.max_steps  = need_long(argc, argv, i);
        else if(a == "--avg-steps")  cfg.avg_steps  = need_long(argc, argv, i);
        else if(a == "--velocity")
        {
            if(i + 1 >= argc) { usage(); die("missing value for --velocity"); }
            cfg.velocity_prefix = argv[++i];
        }
        else if(a == "--validate")   cfg.validate = true;
        else if(a == "--quiet")      cfg.quiet    = true;
        else { usage(); die("unknown option " + a); }
    }

    read_structure(structure);
    build_neighbour_tables();
    allocate();

    const double porosity = double(n_pore)/double(NN);
    const double F = FX_BASE * cfg.ff;

    if(!cfg.quiet)
        cout << "# structure " << structure << "  " << NX << "x" << NY << "x" << NZ
             << "  porosity " << porosity << "  force " << F << "  mu " << MU << endl;

    const Solve rx = solve(F, 0, 0, cfg, "fx");
    const Solve ry = solve(0, F, 0, cfg, "fy");
    const Solve rz = solve(0, 0, F, cfg, "fz");

    // --- assemble K: column j is the response to a force along axis j -------
    double K[3][3];
    for(int c = 0; c < 3; c++)
    {
        K[c][0] = MU * rx.q[c] / F;
        K[c][1] = MU * ry.q[c] / F;
        K[c][2] = MU * rz.q[c] / F;
    }

    const double trace = K[0][0] + K[1][1] + K[2][2];
    const double k_xy = 0.5*(K[0][1] + K[1][0]);
    const double k_xz = 0.5*(K[0][2] + K[2][0]);
    const double k_yz = 0.5*(K[1][2] + K[2][1]);
    const double rec_xy = fabs(K[0][1] - K[1][0]) / fabs(trace);
    const double rec_xz = fabs(K[0][2] - K[2][0]) / fabs(trace);
    const double rec_yz = fabs(K[1][2] - K[2][1]) / fabs(trace);
    const double recip  = max(rec_xy, max(rec_xz, rec_yz));

    double Ks[3][3] = {{K[0][0], k_xy, k_xz},
                       {k_xy, K[1][1], k_yz},
                       {k_xz, k_yz, K[2][2]}};
    double val[3], vec[3][3];
    jacobi3(Ks, val, vec);

    const double kmean = trace/3.0;
    // Fractional anisotropy, the standard scale-free measure for a symmetric
    // positive tensor: 0 = isotropic, 1 = one non-zero principal value.
    const double dv = (val[0]-val[1])*(val[0]-val[1])
                    + (val[1]-val[2])*(val[1]-val[2])
                    + (val[2]-val[0])*(val[2]-val[0]);
    const double nrm = val[0]*val[0] + val[1]*val[1] + val[2]*val[2];
    const double fa  = nrm > 0 ? sqrt(0.5*dv/nrm) : NAN;

    if(cfg.validate)
    {
        const double c = 1.0/sqrt(3.0);
        const Solve rv = solve(F*c, F*c, F*c, cfg, "val");
        double pred[3], err = 0, qn = 0;
        for(int r = 0; r < 3; r++)
        {
            pred[r] = (Ks[r][0]*F*c + Ks[r][1]*F*c + Ks[r][2]*F*c) / MU;
            err += (rv.q[r]-pred[r])*(rv.q[r]-pred[r]);
            qn  += rv.q[r]*rv.q[r];
        }
        cout << setprecision(6);
        cout << "# tensor test along (1,1,1)  (converged " << rv.converged
             << ", " << rv.steps << " steps)" << endl;
        cout << "#   predicted q = (" << pred[0] << ", " << pred[1] << ", " << pred[2] << ")" << endl;
        cout << "#   measured  q = (" << rv.q[0] << ", " << rv.q[1] << ", " << rv.q[2] << ")" << endl;
        cout << "#   relative error " << sqrt(err)/(qn > 0 ? sqrt(qn) : 1) << endl;
    }

    // --- append one row -----------------------------------------------------
    bool need_header = true;
    { ifstream probe(resultfile.c_str(), ios::ate);
      if(probe.good() && probe.tellg() > 0) need_header = false; }

    ofstream csv(resultfile.c_str(), ios::app);
    if(!csv) die("cannot open " + resultfile + " for writing");

    if(need_header)
    {
        csv << "filename,ff,force,mu,nx,ny,nz,porosity,"
               "steps_fx,conv_fx,steps_fy,conv_fy,steps_fz,conv_fz,";
        const char *tg[3] = {"fx","fy","fz"};
        const char *cp[3] = {"x","y","z"};
        for(int r = 0; r < 3; r++) for(int c = 0; c < 3; c++)
            csv << "q" << cp[c] << "_" << tg[r] << ',';
        for(int r = 0; r < 3; r++) for(int c = 0; c < 3; c++)
            csv << "se_q" << cp[c] << "_" << tg[r] << ',';
        for(int r = 0; r < 3; r++) for(int c = 0; c < 3; c++)
            csv << "K_" << cp[r] << cp[c] << ',';
        csv << "k_xy,k_xz,k_yz,"
               "recip_xy,recip_xz,recip_yz,recip_resid,"
               "k1,k2,k3,k_mean,fa,"
               "v1_x,v1_y,v1_z,v2_x,v2_y,v2_z,v3_x,v3_y,v3_z\n";
    }

    csv << setprecision(10);
    csv << structure << ',' << cfg.ff << ',' << F << ',' << MU << ','
        << NX << ',' << NY << ',' << NZ << ',' << porosity << ','
        << rx.steps << ',' << rx.converged << ','
        << ry.steps << ',' << ry.converged << ','
        << rz.steps << ',' << rz.converged << ',';
    const Solve *runs[3] = {&rx, &ry, &rz};
    for(int r = 0; r < 3; r++) for(int c = 0; c < 3; c++) csv << runs[r]->q[c]  << ',';
    for(int r = 0; r < 3; r++) for(int c = 0; c < 3; c++) csv << runs[r]->se[c] << ',';
    for(int r = 0; r < 3; r++) for(int c = 0; c < 3; c++) csv << K[r][c] << ',';
    csv << k_xy << ',' << k_xz << ',' << k_yz << ','
        << rec_xy << ',' << rec_xz << ',' << rec_yz << ',' << recip << ','
        << val[0] << ',' << val[1] << ',' << val[2] << ',' << kmean << ',' << fa << ',';
    for(int j = 0; j < 3; j++) for(int i = 0; i < 3; i++)
        csv << vec[i][j] << (j == 2 && i == 2 ? '\n' : ',');
    csv.close();

    if(!cfg.quiet)
    {
        cout << setprecision(6);
        cout << "# K =" << endl;
        for(int r = 0; r < 3; r++)
            cout << "#   [ " << setw(12) << K[r][0] << " " << setw(12) << K[r][1]
                 << " " << setw(12) << K[r][2] << " ]" << endl;
        cout << "# reciprocity residuals  xy " << rec_xy << "  xz " << rec_xz
             << "  yz " << rec_yz << endl;
        cout << "# k1 " << val[0] << "  k2 " << val[1] << "  k3 " << val[2]
             << "  k_mean " << kmean << "  FA " << fa << endl;
    }

    return (rx.converged && ry.converged && rz.converged) ? 0 : 3;
}
