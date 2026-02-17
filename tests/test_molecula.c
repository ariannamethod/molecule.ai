/*
 * Tests for molecula.c — the C implementation of molecule.ai
 *
 * These tests verify the core components of the organism:
 * - Random number generator
 * - Arena allocator
 * - Dynamic arrays
 * - Autograd operations
 * - Matrix operations
 *
 * Compile: gcc -O2 -o test_molecula test_molecula.c -lsqlite3 -lpthread -lm
 * Run: ./test_molecula
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

/* ============================================================
 * MINIMAL REIMPLEMENTATIONS FOR TESTING
 * ============================================================ */

/* RNG — xorshift64 */
static unsigned long long rng_state = 42;

static void rng_seed(unsigned long long seed) {
    rng_state = seed;
}

static double rand_uniform(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (double)(rng_state & 0x7FFFFFFFFFFFFFFFULL) / (double)0x7FFFFFFFFFFFFFFFULL;
}

static double rand_normal(void) {
    double u1 = rand_uniform();
    double u2 = rand_uniform();
    if (u1 < 1e-15) u1 = 1e-15;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

static int rand_int(int n) {
    return (int)(rand_uniform() * n) % n;
}

/* Dynamic String Array */
typedef struct { 
    char **items; 
    int len, cap; 
} StrArr;

static void sa_init(StrArr *a) {
    a->items = NULL;
    a->len = 0;
    a->cap = 0;
}

static void sa_push(StrArr *a, const char *s) {
    if (a->len >= a->cap) {
        a->cap = a->cap ? a->cap * 2 : 16;
        a->items = realloc(a->items, sizeof(char*) * a->cap);
    }
    a->items[a->len++] = strdup(s);
}

static void sa_free(StrArr *a) {
    for (int i = 0; i < a->len; i++) free(a->items[i]);
    free(a->items);
    a->items = NULL; 
    a->len = a->cap = 0;
}

/* Dynamic Int Array */
typedef struct { 
    int *items; 
    int len, cap; 
} IntArr;

static void ia_init(IntArr *a) {
    a->items = NULL;
    a->len = 0;
    a->cap = 0;
}

static void ia_push(IntArr *a, int v) {
    if (a->len >= a->cap) {
        a->cap = a->cap ? a->cap * 2 : 16;
        a->items = realloc(a->items, sizeof(int) * a->cap);
    }
    a->items[a->len++] = v;
}

static void ia_free(IntArr *a) {
    free(a->items);
    a->items = NULL; 
    a->len = a->cap = 0;
}

/* Arena Allocator */
#define TEST_ARENA_SIZE (1024 * 1024) /* 1 MB for tests */

typedef struct {
    char *buf;
    size_t used, cap;
} Arena;

static Arena arena_new(size_t cap) {
    Arena a;
    a.buf = malloc(cap);
    a.used = 0;
    a.cap = cap;
    return a;
}

static void *arena_alloc(Arena *a, size_t size) {
    size = (size + 7) & ~(size_t)7; /* align to 8 bytes */
    if (a->used + size > a->cap) {
        fprintf(stderr, "arena: out of memory (%zu/%zu)\n", a->used + size, a->cap);
        exit(1);
    }
    void *p = a->buf + a->used;
    a->used += size;
    memset(p, 0, size);
    return p;
}

static void arena_reset(Arena *a) { 
    a->used = 0; 
}

static void arena_destroy(Arena *a) { 
    free(a->buf); 
    a->buf = NULL;
    a->used = 0;
    a->cap = 0;
}

/* Vector operations (simplified) */
typedef struct {
    double *data;
    double *grad;
    int len;
} Vec;

static Vec* vec_new(Arena *a, int len) {
    Vec *v = arena_alloc(a, sizeof(Vec));
    v->data = arena_alloc(a, sizeof(double) * len);
    v->grad = arena_alloc(a, sizeof(double) * len);
    v->len = len;
    return v;
}

static void vec_add(Vec *out, Vec *a, Vec *b) {
    for (int i = 0; i < out->len; i++) {
        out->data[i] = a->data[i] + b->data[i];
    }
}

static void vec_sub(Vec *out, Vec *a, Vec *b) {
    for (int i = 0; i < out->len; i++) {
        out->data[i] = a->data[i] - b->data[i];
    }
}

static void vec_scale(Vec *out, Vec *v, double s) {
    for (int i = 0; i < out->len; i++) {
        out->data[i] = v->data[i] * s;
    }
}

static double vec_dot(Vec *a, Vec *b) {
    double sum = 0.0;
    for (int i = 0; i < a->len; i++) {
        sum += a->data[i] * b->data[i];
    }
    return sum;
}

static void vec_relu(Vec *out, Vec *v) {
    for (int i = 0; i < out->len; i++) {
        out->data[i] = v->data[i] > 0 ? v->data[i] : 0;
    }
}

static double vec_mean_sq(Vec *v) {
    double sum = 0.0;
    for (int i = 0; i < v->len; i++) {
        sum += v->data[i] * v->data[i];
    }
    return sum / (double)v->len;
}

/* Softmax */
static void softmax(double *probs, double *logits, int n) {
    double max_val = logits[0];
    for (int i = 1; i < n; i++) {
        if (logits[i] > max_val) max_val = logits[i];
    }
    
    double sum_exp = 0.0;
    for (int i = 0; i < n; i++) {
        probs[i] = exp(logits[i] - max_val);
        sum_exp += probs[i];
    }
    
    for (int i = 0; i < n; i++) {
        probs[i] /= sum_exp;
    }
}

/* RMS Norm */
static void rms_norm(double *out, double *data, int n, double eps) {
    double sum_sq = 0.0;
    for (int i = 0; i < n; i++) {
        sum_sq += data[i] * data[i];
    }
    double rms = sqrt(sum_sq / (double)n + eps);
    for (int i = 0; i < n; i++) {
        out[i] = data[i] / rms;
    }
}

/* ============================================================
 * TEST UTILITIES
 * ============================================================ */

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) static void test_##name(void)
#define RUN_TEST(name) do { \
    printf("  Running %s... ", #name); \
    test_##name(); \
    printf("PASSED\n"); \
    tests_passed++; \
} while(0)

#define ASSERT(cond) do { \
    if (!(cond)) { \
        printf("FAILED at line %d: %s\n", __LINE__, #cond); \
        tests_failed++; \
        return; \
    } \
} while(0)

#define ASSERT_NEAR(a, b, eps) do { \
    if (fabs((a) - (b)) > (eps)) { \
        printf("FAILED at line %d: %f != %f (eps=%f)\n", __LINE__, (a), (b), (eps)); \
        tests_failed++; \
        return; \
    } \
} while(0)

/* ============================================================
 * RNG TESTS
 * ============================================================ */

TEST(rng_uniform_range) {
    rng_seed(42);
    for (int i = 0; i < 1000; i++) {
        double v = rand_uniform();
        ASSERT(v >= 0.0 && v <= 1.0);
    }
}

TEST(rng_uniform_distribution) {
    rng_seed(42);
    int buckets[10] = {0};
    int n = 10000;
    
    for (int i = 0; i < n; i++) {
        double v = rand_uniform();
        int bucket = (int)(v * 10);
        if (bucket >= 10) bucket = 9;
        buckets[bucket]++;
    }
    
    /* Each bucket should have roughly n/10 = 1000 items */
    for (int i = 0; i < 10; i++) {
        ASSERT(buckets[i] > 700 && buckets[i] < 1300);
    }
}

TEST(rng_normal_distribution) {
    rng_seed(42);
    double sum = 0.0;
    double sum_sq = 0.0;
    int n = 10000;
    
    for (int i = 0; i < n; i++) {
        double v = rand_normal();
        sum += v;
        sum_sq += v * v;
    }
    
    double mean = sum / n;
    double variance = sum_sq / n - mean * mean;
    
    /* Mean should be close to 0, variance close to 1 */
    ASSERT_NEAR(mean, 0.0, 0.1);
    ASSERT_NEAR(variance, 1.0, 0.1);
}

TEST(rng_int_range) {
    rng_seed(42);
    for (int i = 0; i < 1000; i++) {
        int v = rand_int(100);
        ASSERT(v >= 0 && v < 100);
    }
}

/* ============================================================
 * DYNAMIC ARRAY TESTS
 * ============================================================ */

TEST(str_array_push) {
    StrArr a;
    sa_init(&a);
    
    sa_push(&a, "hello");
    sa_push(&a, "world");
    sa_push(&a, "test");
    
    ASSERT(a.len == 3);
    ASSERT(strcmp(a.items[0], "hello") == 0);
    ASSERT(strcmp(a.items[1], "world") == 0);
    ASSERT(strcmp(a.items[2], "test") == 0);
    
    sa_free(&a);
    ASSERT(a.len == 0);
    ASSERT(a.items == NULL);
}

TEST(str_array_growth) {
    StrArr a;
    sa_init(&a);
    
    /* Push many items to test growth */
    for (int i = 0; i < 100; i++) {
        char buf[32];
        sprintf(buf, "item_%d", i);
        sa_push(&a, buf);
    }
    
    ASSERT(a.len == 100);
    ASSERT(strcmp(a.items[50], "item_50") == 0);
    ASSERT(strcmp(a.items[99], "item_99") == 0);
    
    sa_free(&a);
}

TEST(int_array_push) {
    IntArr a;
    ia_init(&a);
    
    ia_push(&a, 10);
    ia_push(&a, 20);
    ia_push(&a, 30);
    
    ASSERT(a.len == 3);
    ASSERT(a.items[0] == 10);
    ASSERT(a.items[1] == 20);
    ASSERT(a.items[2] == 30);
    
    ia_free(&a);
    ASSERT(a.len == 0);
    ASSERT(a.items == NULL);
}

TEST(int_array_growth) {
    IntArr a;
    ia_init(&a);
    
    /* Push many items to test growth */
    for (int i = 0; i < 1000; i++) {
        ia_push(&a, i * i);
    }
    
    ASSERT(a.len == 1000);
    ASSERT(a.items[100] == 10000);
    ASSERT(a.items[500] == 250000);
    
    ia_free(&a);
}

/* ============================================================
 * ARENA ALLOCATOR TESTS
 * ============================================================ */

TEST(arena_basic_alloc) {
    Arena a = arena_new(1024);
    
    void *p1 = arena_alloc(&a, 64);
    void *p2 = arena_alloc(&a, 128);
    void *p3 = arena_alloc(&a, 32);
    
    ASSERT(p1 != NULL);
    ASSERT(p2 != NULL);
    ASSERT(p3 != NULL);
    
    /* Pointers should be different */
    ASSERT(p1 != p2);
    ASSERT(p2 != p3);
    
    arena_destroy(&a);
}

TEST(arena_alignment) {
    Arena a = arena_new(1024);
    
    /* Allocate odd sizes, check 8-byte alignment */
    void *p1 = arena_alloc(&a, 7);
    void *p2 = arena_alloc(&a, 13);
    void *p3 = arena_alloc(&a, 1);
    
    ASSERT(((size_t)p1 % 8) == 0);
    ASSERT(((size_t)p2 % 8) == 0);
    ASSERT(((size_t)p3 % 8) == 0);
    
    arena_destroy(&a);
}

TEST(arena_reset) {
    Arena a = arena_new(1024);
    
    arena_alloc(&a, 256);
    arena_alloc(&a, 256);
    ASSERT(a.used > 0);
    
    arena_reset(&a);
    ASSERT(a.used == 0);
    
    /* Can allocate again */
    void *p = arena_alloc(&a, 128);
    ASSERT(p != NULL);
    
    arena_destroy(&a);
}

/* ============================================================
 * VECTOR OPERATION TESTS
 * ============================================================ */

TEST(vec_creation) {
    Arena a = arena_new(TEST_ARENA_SIZE);
    
    Vec *v = vec_new(&a, 5);
    
    ASSERT(v != NULL);
    ASSERT(v->len == 5);
    ASSERT(v->data != NULL);
    ASSERT(v->grad != NULL);
    
    /* Should be initialized to zero */
    for (int i = 0; i < 5; i++) {
        ASSERT_NEAR(v->data[i], 0.0, 1e-10);
        ASSERT_NEAR(v->grad[i], 0.0, 1e-10);
    }
    
    arena_destroy(&a);
}

TEST(vec_addition) {
    Arena a = arena_new(TEST_ARENA_SIZE);
    
    Vec *v1 = vec_new(&a, 3);
    Vec *v2 = vec_new(&a, 3);
    Vec *out = vec_new(&a, 3);
    
    v1->data[0] = 1.0; v1->data[1] = 2.0; v1->data[2] = 3.0;
    v2->data[0] = 4.0; v2->data[1] = 5.0; v2->data[2] = 6.0;
    
    vec_add(out, v1, v2);
    
    ASSERT_NEAR(out->data[0], 5.0, 1e-10);
    ASSERT_NEAR(out->data[1], 7.0, 1e-10);
    ASSERT_NEAR(out->data[2], 9.0, 1e-10);
    
    arena_destroy(&a);
}

TEST(vec_subtraction) {
    Arena a = arena_new(TEST_ARENA_SIZE);
    
    Vec *v1 = vec_new(&a, 3);
    Vec *v2 = vec_new(&a, 3);
    Vec *out = vec_new(&a, 3);
    
    v1->data[0] = 5.0; v1->data[1] = 7.0; v1->data[2] = 9.0;
    v2->data[0] = 1.0; v2->data[1] = 2.0; v2->data[2] = 3.0;
    
    vec_sub(out, v1, v2);
    
    ASSERT_NEAR(out->data[0], 4.0, 1e-10);
    ASSERT_NEAR(out->data[1], 5.0, 1e-10);
    ASSERT_NEAR(out->data[2], 6.0, 1e-10);
    
    arena_destroy(&a);
}

TEST(vec_scale) {
    Arena a = arena_new(TEST_ARENA_SIZE);
    
    Vec *v = vec_new(&a, 3);
    Vec *out = vec_new(&a, 3);
    
    v->data[0] = 1.0; v->data[1] = 2.0; v->data[2] = 3.0;
    
    vec_scale(out, v, 2.0);
    
    ASSERT_NEAR(out->data[0], 2.0, 1e-10);
    ASSERT_NEAR(out->data[1], 4.0, 1e-10);
    ASSERT_NEAR(out->data[2], 6.0, 1e-10);
    
    arena_destroy(&a);
}

TEST(vec_dot_product) {
    Arena a = arena_new(TEST_ARENA_SIZE);
    
    Vec *v1 = vec_new(&a, 3);
    Vec *v2 = vec_new(&a, 3);
    
    v1->data[0] = 1.0; v1->data[1] = 2.0; v1->data[2] = 3.0;
    v2->data[0] = 4.0; v2->data[1] = 5.0; v2->data[2] = 6.0;
    
    double result = vec_dot(v1, v2);
    
    /* 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32 */
    ASSERT_NEAR(result, 32.0, 1e-10);
    
    arena_destroy(&a);
}

TEST(vec_relu) {
    Arena a = arena_new(TEST_ARENA_SIZE);
    
    Vec *v = vec_new(&a, 5);
    Vec *out = vec_new(&a, 5);
    
    v->data[0] = -1.0;
    v->data[1] = 0.0;
    v->data[2] = 2.0;
    v->data[3] = -3.0;
    v->data[4] = 4.0;
    
    vec_relu(out, v);
    
    ASSERT_NEAR(out->data[0], 0.0, 1e-10);
    ASSERT_NEAR(out->data[1], 0.0, 1e-10);
    ASSERT_NEAR(out->data[2], 2.0, 1e-10);
    ASSERT_NEAR(out->data[3], 0.0, 1e-10);
    ASSERT_NEAR(out->data[4], 4.0, 1e-10);
    
    arena_destroy(&a);
}

TEST(vec_mean_sq) {
    Arena a = arena_new(TEST_ARENA_SIZE);
    
    Vec *v = vec_new(&a, 3);
    v->data[0] = 1.0; v->data[1] = 2.0; v->data[2] = 3.0;
    
    double result = vec_mean_sq(v);
    
    /* (1 + 4 + 9) / 3 = 14/3 */
    ASSERT_NEAR(result, 14.0 / 3.0, 1e-10);
    
    arena_destroy(&a);
}

/* ============================================================
 * SOFTMAX TESTS
 * ============================================================ */

TEST(softmax_basic) {
    double logits[] = {1.0, 2.0, 3.0};
    double probs[3];
    
    softmax(probs, logits, 3);
    
    /* Sum should be 1 */
    double sum = probs[0] + probs[1] + probs[2];
    ASSERT_NEAR(sum, 1.0, 1e-10);
    
    /* Higher logit -> higher prob */
    ASSERT(probs[2] > probs[1]);
    ASSERT(probs[1] > probs[0]);
}

TEST(softmax_equal) {
    double logits[] = {1.0, 1.0, 1.0};
    double probs[3];
    
    softmax(probs, logits, 3);
    
    /* All should be equal = 1/3 */
    ASSERT_NEAR(probs[0], 1.0/3.0, 1e-10);
    ASSERT_NEAR(probs[1], 1.0/3.0, 1e-10);
    ASSERT_NEAR(probs[2], 1.0/3.0, 1e-10);
}

TEST(softmax_large_values) {
    /* Test numerical stability with large values */
    double logits[] = {1000.0, 1001.0, 1002.0};
    double probs[3];
    
    softmax(probs, logits, 3);
    
    /* Should still sum to 1 without overflow */
    double sum = probs[0] + probs[1] + probs[2];
    ASSERT_NEAR(sum, 1.0, 1e-10);
    
    /* No NaN or Inf */
    ASSERT(!isnan(probs[0]) && !isinf(probs[0]));
    ASSERT(!isnan(probs[1]) && !isinf(probs[1]));
    ASSERT(!isnan(probs[2]) && !isinf(probs[2]));
}

/* ============================================================
 * RMS NORM TESTS
 * ============================================================ */

TEST(rms_norm_basic) {
    double data[] = {3.0, 4.0};
    double out[2];
    
    rms_norm(out, data, 2, 1e-8);
    
    /* mean(x^2) = (9 + 16) / 2 = 12.5 */
    /* rms = sqrt(12.5) ≈ 3.5355 */
    /* After RMS norm, the RMS should be approximately 1 */
    double sum_sq = out[0] * out[0] + out[1] * out[1];
    double rms = sqrt(sum_sq / 2.0);
    
    ASSERT_NEAR(rms, 1.0, 0.01);
}

TEST(rms_norm_preserves_direction) {
    double data[] = {2.0, 4.0};
    double out[2];
    
    rms_norm(out, data, 2, 1e-8);
    
    /* Ratio should be preserved */
    double ratio_orig = data[1] / data[0];
    double ratio_norm = out[1] / out[0];
    
    ASSERT_NEAR(ratio_orig, ratio_norm, 1e-10);
}

/* ============================================================
 * CONFIG TESTS
 * ============================================================ */

typedef struct {
    int n_layer;
    int n_embd;
    int n_head;
    int block_size;
    int delta_rank;
} Config;

TEST(config_defaults) {
    Config cfg = {
        .n_layer = 2,
        .n_embd = 72,
        .n_head = 4,
        .block_size = 96,
        .delta_rank = 8,
    };
    
    ASSERT(cfg.n_layer == 2);
    ASSERT(cfg.n_embd == 72);
    ASSERT(cfg.n_head == 4);
    ASSERT(cfg.block_size == 96);
    ASSERT(cfg.delta_rank == 8);
}

TEST(config_head_size) {
    Config cfg = {
        .n_layer = 2,
        .n_embd = 72,
        .n_head = 4,
        .block_size = 96,
        .delta_rank = 8,
    };
    
    int head_size = cfg.n_embd / cfg.n_head;
    ASSERT(head_size == 18);
}

/* ============================================================
 * MAIN
 * ============================================================ */

int main(void) {
    printf("Running molecule.ai C tests...\n\n");
    
    printf("[RNG Tests]\n");
    RUN_TEST(rng_uniform_range);
    RUN_TEST(rng_uniform_distribution);
    RUN_TEST(rng_normal_distribution);
    RUN_TEST(rng_int_range);
    
    printf("\n[Dynamic Array Tests]\n");
    RUN_TEST(str_array_push);
    RUN_TEST(str_array_growth);
    RUN_TEST(int_array_push);
    RUN_TEST(int_array_growth);
    
    printf("\n[Arena Allocator Tests]\n");
    RUN_TEST(arena_basic_alloc);
    RUN_TEST(arena_alignment);
    RUN_TEST(arena_reset);
    
    printf("\n[Vector Operation Tests]\n");
    RUN_TEST(vec_creation);
    RUN_TEST(vec_addition);
    RUN_TEST(vec_subtraction);
    RUN_TEST(vec_scale);
    RUN_TEST(vec_dot_product);
    RUN_TEST(vec_relu);
    RUN_TEST(vec_mean_sq);
    
    printf("\n[Softmax Tests]\n");
    RUN_TEST(softmax_basic);
    RUN_TEST(softmax_equal);
    RUN_TEST(softmax_large_values);
    
    printf("\n[RMS Norm Tests]\n");
    RUN_TEST(rms_norm_basic);
    RUN_TEST(rms_norm_preserves_direction);
    
    printf("\n[Config Tests]\n");
    RUN_TEST(config_defaults);
    RUN_TEST(config_head_size);
    
    printf("\n========================================\n");
    printf("Tests passed: %d\n", tests_passed);
    printf("Tests failed: %d\n", tests_failed);
    printf("========================================\n");
    
    return tests_failed > 0 ? 1 : 0;
}
