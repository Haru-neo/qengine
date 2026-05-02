#include "../src/gguf.h"
#include <cstdio>
#include <algorithm>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s <file.gguf>\n", argv[0]); return 1; }
    GGUFFile g;
    if (!g.open(argv[1])) return 1;

    printf("\n=== Metadata ===\n");
    for (auto& [k, v] : g.meta_str) printf("  str %-40s = %s\n", k.c_str(), v.c_str());
    for (auto& [k, v] : g.meta_u32) printf("  u32 %-40s = %u\n", k.c_str(), v);
    for (auto& [k, v] : g.meta_u64) printf("  u64 %-40s = %lu\n", k.c_str(), v);
    for (auto& [k, v] : g.meta_f32) printf("  f32 %-40s = %g\n", k.c_str(), v);
    for (auto& [k, v] : g.meta_str_arr) printf("  arr %-40s = [%zu strings]\n", k.c_str(), v.size());
    for (auto& [k, v] : g.meta_i32_arr) printf("  arr %-40s = [%zu i32]\n", k.c_str(), v.size());
    for (auto& [k, v] : g.meta_f32_arr) printf("  arr %-40s = [%zu f32]\n", k.c_str(), v.size());

    std::vector<const TensorInfo*> tensors;
    for (auto& [n, t] : g.tensors) tensors.push_back(&t);
    std::sort(tensors.begin(), tensors.end(),
              [](auto a, auto b){ return a->name < b->name; });

    printf("\n=== Tensors (%zu) ===\n", tensors.size());
    for (auto* t : tensors) {
        printf("  %-60s type=%u dims=[%lu,%lu,%lu,%lu] bytes=%lu\n",
               t->name.c_str(), (unsigned)t->type,
               t->dims[0], t->dims[1], t->dims[2], t->dims[3],
               (unsigned long)t->byte_size());
    }
    g.close();
    return 0;
}
