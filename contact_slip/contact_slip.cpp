// Process-local ODE interception: never replaces the installed simulator/library.
#include <ode/ode.h>
#include <dlfcn.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <random>

namespace {
[[noreturn]] void fail(const char* message) {
    std::fprintf(stderr, "CONTACT_SLIP_ERROR %s\n", message);
    std::fflush(stderr);
    std::abort();
}
double number(const char* key) {
    const char* text = std::getenv(key);
    if (!text || !*text) fail(key);
    char* end = nullptr;
    double value = std::strtod(text, &end);
    if (*end || !std::isfinite(value)) fail(key);
    return value;
}
struct State {
    double low, high, mean, stddev;
    bool gaussian, audit;
    FILE* log;
    unsigned long long count = 0;
    std::mt19937_64 rng;
    std::normal_distribution<double> normal;
    std::mutex mutex;
    State(): low(number("CONTACT_LOW")), high(number("CONTACT_HIGH")),
             mean(number("CONTACT_MEAN")), stddev(number("CONTACT_STD")),
             gaussian(false), audit(false), log(nullptr), normal(mean, stddev) {
        if (!(0 <= low && low < high && stddev > 0 && low <= mean && mean <= high))
            fail("invalid distribution parameters");
        const char* mode = std::getenv("CONTACT_MODE");
        if (!mode || (std::strcmp(mode,"constant") && std::strcmp(mode,"gaussian") && std::strcmp(mode,"audit")))
            fail("CONTACT_MODE must be constant, gaussian or audit");
        gaussian = !std::strcmp(mode,"gaussian");
        audit = !std::strcmp(mode,"audit");
        double seed = number("CONTACT_SEED");
        if (seed < 0 || seed > 4294967295. || seed != std::floor(seed)) fail("invalid seed");
        rng.seed(static_cast<unsigned long long>(seed));
        const char* path = std::getenv("CONTACT_TRACE");
        if (!path || !(log = std::fopen(path, "wx"))) fail("trace must be a new writable file");
#ifdef dDOUBLE
        if (!dCheckConfiguration("ODE_double_precision")) fail("ODE precision mismatch");
#else
        if (!dCheckConfiguration("ODE_single_precision")) fail("ODE precision mismatch");
#endif
        std::fprintf(log,"contact,x,y,z,original_slip1,original_slip2,applied_slip,original_mode,applied_mode,slip1_enabled,slip2_enabled\n");
        std::fflush(log);
        std::fprintf(stdout,"CONTACT_SLIP_READY mode=%s seed=%.0f low=%.17g high=%.17g mean=%.17g std=%.17g\n",
                     mode, seed, low, high, mean, stddev);
        std::fflush(stdout);
    }
    ~State() { if (log) std::fclose(log); }
};
State& state() { static State value; return value; }
bool ground(dGeomID geom) {
    if (!geom || dGeomGetClass(geom) != dPlaneClass) return false;
    dVector4 plane;
    dGeomPlaneGetParams(geom, plane);
    return std::abs(plane[0]) < 1e-6 && std::abs(plane[1]) < 1e-6 &&
           std::abs(std::abs(plane[2])-1.) < 1e-6 && std::abs(plane[3]) < 1e-6;
}
}

__attribute__((constructor)) static void initialize_contact_slip() { state(); }

extern "C" dJointID dJointCreateContact(dWorldID world, dJointGroupID group, const dContact* input) {
    using Function = dJointID (*)(dWorldID, dJointGroupID, const dContact*);
    static auto real = reinterpret_cast<Function>(dlsym(RTLD_NEXT, "dJointCreateContact"));
    if (!real) fail("cannot resolve original dJointCreateContact");
    // Soccer field extent, excluding walls and out-of-bounds ground contacts.
    if (!input || !(ground(input->geom.g1) || ground(input->geom.g2)) ||
        std::abs(input->geom.pos[0]) > 15. || std::abs(input->geom.pos[1]) > 10.)
        return real(world, group, input);
    State& s = state();
    std::lock_guard<std::mutex> guard(s.mutex);
    double sample = s.mean;
    if (s.gaussian) {
        unsigned int attempts = 0;
        do {
            if (++attempts > 1000000) fail("rejection sampler exhausted");
            sample = s.normal(s.rng);
        } while (sample < s.low || sample > s.high);
    }
    dContact contact = *input;
    if (!s.audit) {
        contact.surface.mode |= dContactSlip1 | dContactSlip2;
        contact.surface.slip1 = contact.surface.slip2 = sample;
    }
    // Log solver-bound parameters, not merely the requested scene configuration.
    std::fprintf(s.log,"%llu,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%d,%d,%d,%d\n", ++s.count,
                 double(contact.geom.pos[0]), double(contact.geom.pos[1]), double(contact.geom.pos[2]),
                 double(input->surface.slip1), double(input->surface.slip2), double(contact.surface.slip1),
                 input->surface.mode, contact.surface.mode,
                 bool(contact.surface.mode & dContactSlip1), bool(contact.surface.mode & dContactSlip2));
    std::fflush(s.log);  // retain evidence even on native abort or SIGTERM
    return real(world, group, s.audit ? input : &contact);
}
