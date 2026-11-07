/* Single-step inference kernel for ScaledWunderGRU (no MoE).
 * One call per data point: features -> standardize -> input proj + LN -> GRU stack -> head.
 * Supports an ensemble of K models (predictions averaged); each model reads one feature set:
 *   F=0  "v144"       stationary 144 features (src/features.py)
 *   F=1  "raw_ev_ema" 456 features (src/feature_sets.py)
 * Each feature set is computed once per step, only if some model uses it.
 * Compiled at model load time: gcc -O3 -march=native -shared -fPIC.
 */
#include <math.h>
#include <string.h>

#define NRAW 112
#define NF 144
#define NEV 456
#define MAXF 512
#define MAXH 512
#define MAXL 4

typedef struct {
    int H, L, F, nf;                  /* F = feature set id, nf = its width */
    const float *mean, *std;          /* nf */
    const float *Wp, *bp;             /* H x nf, H */
    const float *lnp_g, *lnp_b;       /* H */
    const float *Wih[MAXL], *Whh[MAXL], *bih[MAXL], *bhh[MAXL]; /* 3H x H */
    const float *ln_g[MAXL], *ln_b[MAXL];
    const float *lnh_g, *lnh_b;       /* head LN */
    const float *W1, *b1;             /* H x H */
    const float *W2, *b2;             /* 2 x H */
    float *h;                         /* L x H recurrent state */
} Model;

typedef struct {
    int started;
    double prev_mid0, prev_mid1;
    double E[13];
} FeatState;

static const double ALPHA[13] = {0.2, 0.05, 0.01, 0.2, 0.05, 0.01, 0.05, 0.05, 0.05, 0.05, 0.05, 0.002, 0.05};

static void features(const float *x, FeatState *s, float *f) {
    double sb0 = 0, sa0 = 0, sb1 = 0, sa1 = 0, qb0 = 0, qa0 = 0;
    for (int i = 0; i < 11; i++) {
        sb0 += x[i]; sa0 += x[11 + i]; sb1 += x[52 + i]; sa1 += x[63 + i];
        qb0 += (double)x[i] * x[i]; qa0 += (double)x[11 + i] * x[11 + i];
    }
    double pb0 = sb0 / 11.0, pa0 = sa0 / 11.0, pb1 = sb1 / 11.0, pa1 = sa1 / 11.0;
    double mid0 = 0.5 * (pb0 + pa0), mid1 = 0.5 * (pb1 + pa1);
    double ofi0 = 0, ofi1 = 0;
    for (int i = 0; i < 11; i++) { ofi0 += x[22 + i] - x[33 + i]; ofi1 += x[74 + i] - x[85 + i]; }
    double dp = 0, dv = 0, mf = 0;
    for (int i = 0; i < 4; i++) { dp += x[44 + i]; dv += x[48 + i]; mf += (double)x[44 + i] * x[48 + i]; }
    double l8 = x[22] - x[33], l9 = x[24] - x[35], l10 = x[26] - x[37], l11 = x[32] - x[43];
    double vb = qb0 / 11.0 - pb0 * pb0; if (vb < 0) vb = 0;
    double va = qa0 / 11.0 - pa0 * pa0; if (va < 0) va = 0;
    double r0 = 0, r1 = 0;
    if (s->started) { r0 = mid0 - s->prev_mid0; r1 = mid1 - s->prev_mid1; }
    double u[13] = {r0, r0, r0, ofi0, ofi0, ofi0, r1, mid0 - mid1, r0 * r0, dv, mf, r0, ofi1};
    if (!s->started) { for (int k = 0; k < 13; k++) s->E[k] = u[k]; s->started = 1; }
    else { for (int k = 0; k < 13; k++) s->E[k] += ALPHA[k] * (u[k] - s->E[k]); }
    s->prev_mid0 = mid0; s->prev_mid1 = mid1;
    const double *E = s->E;
    for (int i = 0; i < NRAW; i++) f[i] = x[i];
    for (int i = 0; i < 22; i++) f[i] = (float)(x[i] - (float)mid0);
    for (int i = 52; i < 74; i++) f[i] = (float)(x[i] - (float)mid1);
    double eng[32] = {
        pa0 - pb0, sqrt(vb + 1e-6), sqrt(va + 1e-6), pa0 - pb0, ofi0, l8, l9, l10, l11, pa1 - pb1,
        mid0 - mid1, ofi0 - ofi1, dp, dv, mf, r0,
        E[0], E[1], E[2], E[0] - E[1], E[3], E[4], E[5], E[3] - E[5], E[6], E[6] - E[1],
        E[7], E[8], E[9], E[10], E[11], E[12] - E[4]};
    for (int i = 0; i < 32; i++) f[NRAW + i] = (float)eng[i];
}

/* raw_ev_ema: raw 112, then per event block (changed flag, log1p(steps since change)),
 * one-step diffs, and x - EMA(x) for alpha 0.05 and 0.005 (EMA starts at the first row). */
static const int EV_BLOCKS[4][2] = {{0, 22}, {52, 74}, {22, 44}, {74, 96}};
static const double EV_ALPHA[2] = {0.05, 0.005};

typedef struct {
    int started;
    long step, last[4];
    float prev[NRAW];
    double E[2][NRAW];
} EvState;

static void ev_features(const float *x, EvState *s, float *f) {
    for (int i = 0; i < NRAW; i++) f[i] = x[i];
    for (int b = 0; b < 4; b++) {
        int changed = !s->started;
        for (int i = EV_BLOCKS[b][0]; i < EV_BLOCKS[b][1] && !changed; i++) changed = x[i] != s->prev[i];
        if (changed) s->last[b] = s->step;
        f[NRAW + 2 * b] = (float)changed;
        f[NRAW + 2 * b + 1] = (float)log1p((double)(s->step - s->last[b]));
    }
    float *d = f + NRAW + 8;
    for (int i = 0; i < NRAW; i++) d[i] = s->started ? (float)((double)x[i] - (double)s->prev[i]) : 0.0f;
    for (int k = 0; k < 2; k++) {
        float *dev = f + NRAW + 8 + NRAW * (k + 1);
        for (int i = 0; i < NRAW; i++) {
            if (!s->started) s->E[k][i] = x[i];
            else s->E[k][i] += EV_ALPHA[k] * ((double)x[i] - s->E[k][i]);
            dev[i] = (float)((double)x[i] - s->E[k][i]);
        }
    }
    for (int i = 0; i < NRAW; i++) s->prev[i] = x[i];
    s->started = 1;
    s->step++;
}

static inline float sigm(float v) { return 1.0f / (1.0f + expf(-v)); }

/* WT is the TRANSPOSED weight (cols x rows, row-major), so the inner loop is an axpy over
 * contiguous outputs with no reduction: gcc vectorizes it without -ffast-math. */
static void matvec(const float *restrict WT, const float *restrict b, const float *restrict x,
                   int rows, int cols, float *restrict y) {
    if (b) { for (int r = 0; r < rows; r++) y[r] = b[r]; }
    else { for (int r = 0; r < rows; r++) y[r] = 0.0f; }
    for (int c = 0; c < cols; c++) {
        const float xc = x[c];
        const float *restrict w = WT + (long)c * rows;
        for (int r = 0; r < rows; r++) y[r] += w[r] * xc;
    }
}

static void layernorm(float *v, int n, const float *g, const float *b) {
    float m = 0, var = 0;
    for (int i = 0; i < n; i++) m += v[i];
    m /= n;
    for (int i = 0; i < n; i++) { float d = v[i] - m; var += d * d; }
    var /= n;
    float inv = 1.0f / sqrtf(var + 1e-5f);
    for (int i = 0; i < n; i++) v[i] = (v[i] - m) * inv * g[i] + b[i];
}

static void model_step(Model *M, const float *f, float *out) {
    int H = M->H;
    float z[MAXF], cur[MAXH], gi[3 * MAXH], gh[3 * MAXH], prev[MAXH], t[MAXH];
    for (int i = 0; i < M->nf; i++) z[i] = (f[i] - M->mean[i]) / M->std[i];
    matvec(M->Wp, M->bp, z, H, M->nf, cur);
    layernorm(cur, H, M->lnp_g, M->lnp_b);
    for (int l = 0; l < M->L; l++) {
        float *h = M->h + l * H;
        matvec(M->Wih[l], M->bih[l], cur, 3 * H, H, gi);
        matvec(M->Whh[l], M->bhh[l], h, 3 * H, H, gh);
        for (int i = 0; i < H; i++) {
            float r = sigm(gi[i] + gh[i]);
            float zz = sigm(gi[H + i] + gh[H + i]);
            float n = tanhf(gi[2 * H + i] + r * gh[2 * H + i]);
            h[i] = (1.0f - zz) * n + zz * h[i];
        }
        if (l == 0) { for (int i = 0; i < H; i++) cur[i] = h[i]; }
        else { for (int i = 0; i < H; i++) cur[i] = h[i] + cur[i]; }
        layernorm(cur, H, M->ln_g[l], M->ln_b[l]);
    }
    for (int i = 0; i < H; i++) prev[i] = cur[i];
    layernorm(prev, H, M->lnh_g, M->lnh_b);
    matvec(M->W1, M->b1, prev, H, H, t);
    for (int i = 0; i < H; i++) t[i] = 0.5f * t[i] * (1.0f + erff(t[i] * 0.70710678118f));
    matvec(M->W2, M->b2, t, 2, H, out);
}

/* BranchGRU: 4 input-specialist GRUs (each reads its own columns of raw_ev_ema) + fusion GRU + head. */
#define NBR 4
typedef struct {
    int H, n[NBR];
    const int *idx[NBR];
    const float *mean, *std;
    const float *Wp[NBR], *bp[NBR], *lnp_g[NBR], *lnp_b[NBR];
    const float *Wih[NBR + 1], *Whh[NBR + 1], *bih[NBR + 1], *bhh[NBR + 1];
    const float *ln_g[NBR + 1], *ln_b[NBR + 1];
    const float *Wf, *bf, *lnf_g, *lnf_b;
    const float *lnh_g, *lnh_b, *W1, *b1, *W2, *b2;
    float *h;                         /* (NBR + 1) x H */
} BranchModel;

/* One GRU update of h from z, then LN(h + z) written to out. */
static void gru_res_step(const BranchModel *M, int g, float *h, const float *z, float *out) {
    int H = M->H;
    float gi[3 * MAXH], gh[3 * MAXH];
    matvec(M->Wih[g], M->bih[g], z, 3 * H, H, gi);
    matvec(M->Whh[g], M->bhh[g], h, 3 * H, H, gh);
    for (int i = 0; i < H; i++) {
        float r = sigm(gi[i] + gh[i]);
        float zz = sigm(gi[H + i] + gh[H + i]);
        float n = tanhf(gi[2 * H + i] + r * gh[2 * H + i]);
        h[i] = (1.0f - zz) * n + zz * h[i];
        out[i] = h[i] + z[i];
    }
    layernorm(out, H, M->ln_g[g], M->ln_b[g]);
}

static void branch_step(BranchModel *M, const float *f, float *out) {
    int H = M->H;
    float sel[MAXF], z[MAXH], concat[NBR * MAXH], cur[MAXH], t[MAXH];
    for (int b = 0; b < NBR; b++) {
        for (int i = 0; i < M->n[b]; i++) { int c = M->idx[b][i]; sel[i] = (f[c] - M->mean[c]) / M->std[c]; }
        matvec(M->Wp[b], M->bp[b], sel, H, M->n[b], z);
        layernorm(z, H, M->lnp_g[b], M->lnp_b[b]);
        gru_res_step(M, b, M->h + b * H, z, concat + b * H);
    }
    matvec(M->Wf, M->bf, concat, H, NBR * H, z);
    layernorm(z, H, M->lnf_g, M->lnf_b);
    gru_res_step(M, NBR, M->h + NBR * H, z, cur);
    layernorm(cur, H, M->lnh_g, M->lnh_b);
    matvec(M->W1, M->b1, cur, H, H, t);
    for (int i = 0; i < H; i++) t[i] = 0.5f * t[i] * (1.0f + erff(t[i] * 0.70710678118f));
    matvec(M->W2, M->b2, t, 2, H, out);
}

/* Public API ---------------------------------------------------------------------------- */
#define MAXK 8
static Model MODELS[MAXK];
static int K = 0;
static BranchModel BRANCH[MAXK];
static int KB = 0;
static FeatState FS;
static EvState ES;
static float FEAT[NF], FEAT_EV[NEV];
static int NEED[2];

/* params: flat array of pointers in a fixed order (see cstep.py) */
int wnn_add_model(int H, int L, int F, const float **p, float *hstate) {
    if (K >= MAXK || H > MAXH || L > MAXL || F < 0 || F > 1) return -1;
    Model *M = &MODELS[K];
    int i = 0;
    M->H = H; M->L = L; M->F = F; M->nf = F == 0 ? NF : NEV;
    NEED[F] = 1;
    M->mean = p[i++]; M->std = p[i++]; M->Wp = p[i++]; M->bp = p[i++]; M->lnp_g = p[i++]; M->lnp_b = p[i++];
    for (int l = 0; l < L; l++) {
        M->Wih[l] = p[i++]; M->Whh[l] = p[i++]; M->bih[l] = p[i++]; M->bhh[l] = p[i++];
        M->ln_g[l] = p[i++]; M->ln_b[l] = p[i++];
    }
    M->lnh_g = p[i++]; M->lnh_b = p[i++]; M->W1 = p[i++]; M->b1 = p[i++]; M->W2 = p[i++]; M->b2 = p[i++];
    M->h = hstate;
    return K++;
}

/* Branch models always read raw_ev_ema. Parameter order: see branch_param_order in solution.py. */
int wnn_add_branch(int H, const float **p, const int **idx, const int *n, float *hstate) {
    if (KB >= MAXK || H > MAXH) return -1;
    BranchModel *M = &BRANCH[KB];
    int i = 0;
    M->H = H;
    NEED[1] = 1;
    M->mean = p[i++]; M->std = p[i++];
    for (int b = 0; b < NBR; b++) {
        if (n[b] > MAXF) return -1;
        M->n[b] = n[b]; M->idx[b] = idx[b];
        M->Wp[b] = p[i++]; M->bp[b] = p[i++]; M->lnp_g[b] = p[i++]; M->lnp_b[b] = p[i++];
        M->Wih[b] = p[i++]; M->Whh[b] = p[i++]; M->bih[b] = p[i++]; M->bhh[b] = p[i++];
        M->ln_g[b] = p[i++]; M->ln_b[b] = p[i++];
    }
    M->Wf = p[i++]; M->bf = p[i++]; M->lnf_g = p[i++]; M->lnf_b = p[i++];
    M->Wih[NBR] = p[i++]; M->Whh[NBR] = p[i++]; M->bih[NBR] = p[i++]; M->bhh[NBR] = p[i++];
    M->ln_g[NBR] = p[i++]; M->ln_b[NBR] = p[i++];
    M->lnh_g = p[i++]; M->lnh_b = p[i++]; M->W1 = p[i++]; M->b1 = p[i++]; M->W2 = p[i++]; M->b2 = p[i++];
    M->h = hstate;
    return KB++;
}

void wnn_reset(void) {
    memset(&FS, 0, sizeof(FS));
    memset(&ES, 0, sizeof(ES));
    for (int k = 0; k < K; k++) memset(MODELS[k].h, 0, sizeof(float) * MODELS[k].H * MODELS[k].L);
    for (int k = 0; k < KB; k++) memset(BRANCH[k].h, 0, sizeof(float) * BRANCH[k].H * (NBR + 1));
}

void wnn_step(const float *x, float *out) {
    if (NEED[0]) features(x, &FS, FEAT);
    if (NEED[1]) ev_features(x, &ES, FEAT_EV);
    float acc0 = 0, acc1 = 0, o[2];
    for (int k = 0; k < K; k++) {
        model_step(&MODELS[k], MODELS[k].F == 0 ? FEAT : FEAT_EV, o);
        acc0 += o[0]; acc1 += o[1];
    }
    for (int k = 0; k < KB; k++) {
        branch_step(&BRANCH[k], FEAT_EV, o);
        acc0 += o[0]; acc1 += o[1];
    }
    out[0] = acc0 / (K + KB); out[1] = acc1 / (K + KB);
}

void wnn_features(const float *x, float *f) { features(x, &FS, f); }
void wnn_ev_features(const float *x, float *f) { ev_features(x, &ES, f); }
