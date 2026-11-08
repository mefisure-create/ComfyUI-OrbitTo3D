#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polygon_mesh_processing/triangulate_hole.h>
#include <CGAL/boost/graph/helpers.h>
#include <CGAL/utility.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace PMP = CGAL::Polygon_mesh_processing;
namespace params = CGAL::parameters;
using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point = Kernel::Point_3;
using Triangle = CGAL::Triple<int, int, int>;

static constexpr char MAGIC_IN[8]  = {'O','3','D','P','M','P','0','1'};
static constexpr char MAGIC_OUT[8] = {'O','3','D','P','M','P','O','1'};
static constexpr std::uint32_t VERSION = 2;
static constexpr std::uint32_t FLAG_THIRD_POINTS = 1u;

struct LoopInput {
    std::uint64_t hole_index = 0;
    std::vector<std::uint64_t> global_ids;
    std::vector<Point> points;
    std::vector<Point> third_points;
    bool has_third_points = false;
};

struct LoopResult {
    std::uint64_t hole_index = 0;
    std::uint64_t edges = 0;
    std::int32_t status = 0;
    double elapsed = 0.0;
    std::vector<std::array<std::uint64_t, 3>> faces;
};

template <class T>
void read_exact(std::istream& in, T& value) {
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) throw std::runtime_error("truncated input");
}

template <class T>
void write_exact(std::ostream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!out) throw std::runtime_error("output write failed");
}

void read_bytes(std::istream& in, char* dst, std::size_t bytes) {
    if (bytes == 0) return;
    in.read(dst, static_cast<std::streamsize>(bytes));
    if (!in) throw std::runtime_error("truncated input payload");
}

std::vector<Point> read_points(std::istream& in, std::size_t n) {
    std::vector<double> xyz(n * 3);
    read_bytes(in, reinterpret_cast<char*>(xyz.data()), xyz.size() * sizeof(double));

    std::vector<Point> points;
    points.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        points.emplace_back(xyz[3*i], xyz[3*i+1], xyz[3*i+2]);
    }
    return points;
}

std::vector<LoopInput> read_input(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open input: " + path);

    char magic[8];
    in.read(magic, 8);
    if (!in || !std::equal(std::begin(magic), std::end(magic), std::begin(MAGIC_IN)))
        throw std::runtime_error("bad input magic");

    std::uint32_t version = 0;
    std::uint64_t count = 0;
    read_exact(in, version);
    read_exact(in, count);
    if (version != VERSION) throw std::runtime_error("unsupported input version");

    std::vector<LoopInput> loops;
    loops.reserve(static_cast<std::size_t>(count));

    for (std::uint64_t li = 0; li < count; ++li) {
        LoopInput rec;
        std::uint64_t n64 = 0;
        std::uint32_t flags = 0;
        read_exact(in, rec.hole_index);
        read_exact(in, n64);
        read_exact(in, flags);

        if (n64 < 3) throw std::runtime_error("boundary loop with fewer than 3 vertices");
        const auto n = static_cast<std::size_t>(n64);

        rec.global_ids.resize(n);
        read_bytes(in, reinterpret_cast<char*>(rec.global_ids.data()),
                   n * sizeof(std::uint64_t));

        rec.points = read_points(in, n);
        rec.has_third_points = (flags & FLAG_THIRD_POINTS) != 0;
        if (rec.has_third_points)
            rec.third_points = read_points(in, n);

        loops.push_back(std::move(rec));
    }
    return loops;
}

auto hole_params() {
    return params::use_2d_constrained_delaunay_triangulation(true)
        .use_delaunay_triangulation(true)
        .do_not_use_cubic_algorithm(true);
}

LoopResult triangulate_one(const LoopInput& inrec) {
    LoopResult out;
    out.hole_index = inrec.hole_index;
    out.edges = static_cast<std::uint64_t>(inrec.points.size());
    const auto t0 = std::chrono::steady_clock::now();

    try {
        std::vector<Triangle> patch;
        patch.reserve(inrec.points.size() >= 2 ? inrec.points.size() - 2 : 0);

        if (inrec.has_third_points) {
            PMP::triangulate_hole_polyline(
                inrec.points,
                inrec.third_points,
                std::back_inserter(patch),
                hole_params()
            );
        } else {
            PMP::triangulate_hole_polyline(
                inrec.points,
                std::back_inserter(patch),
                hole_params()
            );
        }

        if (patch.empty()) {
            out.status = 1;
        } else {
            out.faces.reserve(patch.size());
            for (const auto& tri : patch) {
                const int a = tri.first;
                const int b = tri.second;
                const int c = tri.third;
                if (a < 0 || b < 0 || c < 0 ||
                    static_cast<std::size_t>(a) >= inrec.global_ids.size() ||
                    static_cast<std::size_t>(b) >= inrec.global_ids.size() ||
                    static_cast<std::size_t>(c) >= inrec.global_ids.size()) {
                    throw std::runtime_error("CGAL returned out-of-range local index");
                }
                out.faces.push_back({
                    inrec.global_ids[static_cast<std::size_t>(a)],
                    inrec.global_ids[static_cast<std::size_t>(b)],
                    inrec.global_ids[static_cast<std::size_t>(c)]
                });
            }
            out.status = 0;
        }
    } catch (const std::exception& e) {
        out.status = 2;
        std::cerr << "[Orbit3D PMP helper] hole=" << inrec.hole_index
                  << " exception=" << e.what() << std::endl;
    } catch (...) {
        out.status = 2;
        std::cerr << "[Orbit3D PMP helper] hole=" << inrec.hole_index
                  << " unknown exception" << std::endl;
    }

    out.elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0
    ).count();
    return out;
}

void write_output(const std::string& path, const std::vector<LoopResult>& results) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot open output: " + path);

    out.write(MAGIC_OUT, 8);
    write_exact(out, VERSION);
    const std::uint64_t count = static_cast<std::uint64_t>(results.size());
    write_exact(out, count);

    for (const auto& rec : results) {
        write_exact(out, rec.hole_index);
        write_exact(out, rec.edges);
        write_exact(out, rec.status);
        const std::uint64_t tri_count = static_cast<std::uint64_t>(rec.faces.size());
        write_exact(out, tri_count);
        write_exact(out, rec.elapsed);
        for (const auto& tri : rec.faces) {
            write_exact(out, tri[0]);
            write_exact(out, tri[1]);
            write_exact(out, tri[2]);
        }
    }
    out.flush();
    if (!out) throw std::runtime_error("output flush failed");
}

int main(int argc, char** argv) {
    if (argc < 3 || argc > 4) {
        std::cerr << "usage: orbit3d_pmp_polyline INPUT.bin OUTPUT.bin [threads]\n";
        return 2;
    }

    try {
        int threads = 1;
        if (argc == 4) threads = std::max(1, std::stoi(argv[3]));

#ifdef _OPENMP
        omp_set_num_threads(threads);
#else
        (void)threads;
#endif

        auto loops = read_input(argv[1]);
        std::vector<LoopResult> results(loops.size());

        std::cerr << "[Orbit3D PMP helper] version=2 loops=" << loops.size()
                  << " threads=" << threads
#ifdef _OPENMP
                  << " openmp=yes"
#else
                  << " openmp=no"
#endif
                  << std::endl;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 8)
#endif
        for (std::int64_t i = 0;
             i < static_cast<std::int64_t>(loops.size()); ++i) {
            results[static_cast<std::size_t>(i)] =
                triangulate_one(loops[static_cast<std::size_t>(i)]);
        }

        std::size_t ok = 0, empty = 0, except = 0;
        double sum_s = 0.0, max_s = 0.0;
        for (const auto& r : results) {
            if (r.status == 0) ++ok;
            else if (r.status == 1) ++empty;
            else ++except;
            sum_s += r.elapsed;
            max_s = std::max(max_s, r.elapsed);
        }

        write_output(argv[2], results);

        std::cerr << "[Orbit3D PMP helper] done ok=" << ok
                  << " empty=" << empty
                  << " exception=" << except
                  << " cpu_loop_seconds_sum=" << std::fixed << std::setprecision(3)
                  << sum_s << " max_single=" << max_s << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "[Orbit3D PMP helper] FATAL: " << e.what() << std::endl;
        return 1;
    }
}
