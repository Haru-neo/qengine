#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

// GGUF 포맷 상수
#define GGUF_MAGIC 0x46554747  // "GGUF"

enum gguf_type : uint32_t {
    GGUF_TYPE_UINT8   = 0,  GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,  GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,  GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,  GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,  GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10, GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

// 양자화 타입 (Q5_K_M 등)
enum ggml_type : uint32_t {
    GGML_TYPE_F32  = 0,  GGML_TYPE_F16  = 1,
    GGML_TYPE_Q4_0 = 2,  GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q5_0 = 6,  GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,  GGML_TYPE_Q8_1 = 9,
    GGML_TYPE_Q2_K = 10, GGML_TYPE_Q3_K = 11,
    GGML_TYPE_Q4_K = 12, GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14, GGML_TYPE_Q8_K = 15,
    GGML_TYPE_BF16 = 30,
};

struct GGUFString {
    uint64_t len;
    // data follows
};

struct TensorInfo {
    std::string name;
    uint32_t n_dims;
    uint64_t dims[4];
    ggml_type type;
    uint64_t offset;  // from start of data section
    void* data;       // pointer into mmap
    
    uint64_t num_elements() const {
        uint64_t n = 1;
        for (uint32_t i = 0; i < n_dims; i++) n *= dims[i];
        return n;
    }
    
    uint64_t byte_size() const;
};

struct GGUFFile {
    int fd = -1;
    void* mmap_addr = nullptr;
    size_t mmap_size = 0;
    uint8_t* data_start = nullptr;
    
    // Metadata
    std::unordered_map<std::string, std::string> meta_str;
    std::unordered_map<std::string, uint32_t> meta_u32;
    std::unordered_map<std::string, uint64_t> meta_u64;
    std::unordered_map<std::string, float> meta_f32;
    std::unordered_map<std::string, std::vector<std::string>> meta_str_arr;
    std::unordered_map<std::string, std::vector<int32_t>> meta_i32_arr;
    std::unordered_map<std::string, std::vector<uint8_t>> meta_bool_arr;
    std::unordered_map<std::string, std::vector<float>> meta_f32_arr;
    
    // Tensors
    std::unordered_map<std::string, TensorInfo> tensors;
    
    bool open(const char* path) {
        fd = ::open(path, O_RDONLY);
        if (fd < 0) { perror("open"); return false; }
        
        struct stat st;
        fstat(fd, &st);
        mmap_size = st.st_size;

        // Hint to the kernel that we will read sequentially. The loader
        // fan-out across GPU threads otherwise looks random to the disk
        // (each thread reads a different range of the file at the same
        // time), defeating the kernel read-ahead and capping us to ~50
        // MB/s per thread. fadvise on the fd reaches the page cache
        // prefetcher; madvise on the mapping reaches the VMA hinting
        // code. Both are best-effort.
        posix_fadvise(fd, 0, mmap_size, POSIX_FADV_SEQUENTIAL);

        mmap_addr = mmap(nullptr, mmap_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mmap_addr == MAP_FAILED) { perror("mmap"); return false; }
        madvise(mmap_addr, mmap_size, MADV_SEQUENTIAL);
        
        uint8_t* p = (uint8_t*)mmap_addr;
        uint8_t* end = p + mmap_size;
        
        // Header
        uint32_t magic = *(uint32_t*)p; p += 4;
        if (magic != GGUF_MAGIC) {
            fprintf(stderr, "Not a GGUF file (magic: 0x%08x)\n", magic);
            return false;
        }
        uint32_t version = *(uint32_t*)p; p += 4;
        uint64_t n_tensors = *(uint64_t*)p; p += 8;
        uint64_t n_kv = *(uint64_t*)p; p += 8;
        
        printf("GGUF v%u: %lu tensors, %lu metadata\n", version, n_tensors, n_kv);
        
        // Parse metadata
        for (uint64_t i = 0; i < n_kv; i++) {
            p = parse_kv(p);
            if (!p) return false;
        }
        
        // Parse tensor infos
        std::vector<TensorInfo> tinfos(n_tensors);
        for (uint64_t i = 0; i < n_tensors; i++) {
            auto& ti = tinfos[i];
            uint64_t name_len = *(uint64_t*)p; p += 8;
            ti.name = std::string((char*)p, name_len); p += name_len;
            ti.n_dims = *(uint32_t*)p; p += 4;
            for (uint32_t d = 0; d < ti.n_dims; d++) {
                ti.dims[d] = *(uint64_t*)p; p += 8;
            }
            for (uint32_t d = ti.n_dims; d < 4; d++) ti.dims[d] = 1;
            ti.type = *(ggml_type*)p; p += 4;
            ti.offset = *(uint64_t*)p; p += 8;
        }
        
        // Data section (aligned to 32 bytes)
        size_t header_size = p - (uint8_t*)mmap_addr;
        size_t alignment = 32;
        size_t data_offset = (header_size + alignment - 1) & ~(alignment - 1);
        data_start = (uint8_t*)mmap_addr + data_offset;
        
        for (auto& ti : tinfos) {
            ti.data = data_start + ti.offset;
            tensors[ti.name] = ti;
        }
        
        return true;
    }
    
    void close() {
        if (mmap_addr && mmap_addr != MAP_FAILED) munmap(mmap_addr, mmap_size);
        if (fd >= 0) ::close(fd);
    }
    
    // Helpers
    std::string get_str(const std::string& key, const std::string& def = "") const {
        auto it = meta_str.find(key);
        return it != meta_str.end() ? it->second : def;
    }
    uint32_t get_u32(const std::string& key, uint32_t def = 0) const {
        auto it = meta_u32.find(key);
        return it != meta_u32.end() ? it->second : def;
    }
    float get_f32(const std::string& key, float def = 0) const {
        auto it = meta_f32.find(key);
        return it != meta_f32.end() ? it->second : def;
    }
    
    TensorInfo* get_tensor(const std::string& name) {
        auto it = tensors.find(name);
        return it != tensors.end() ? &it->second : nullptr;
    }

private:
    uint8_t* parse_kv(uint8_t* p) {
        uint64_t key_len = *(uint64_t*)p; p += 8;
        std::string key((char*)p, key_len); p += key_len;
        uint32_t vtype = *(uint32_t*)p; p += 4;
        
        switch (vtype) {
            case GGUF_TYPE_STRING: {
                uint64_t slen = *(uint64_t*)p; p += 8;
                meta_str[key] = std::string((char*)p, slen); p += slen;
                break;
            }
            case GGUF_TYPE_UINT32:
                meta_u32[key] = *(uint32_t*)p; p += 4; break;
            case GGUF_TYPE_INT32:
                meta_u32[key] = *(uint32_t*)p; p += 4; break;
            case GGUF_TYPE_UINT64:
                meta_u64[key] = *(uint64_t*)p; p += 8; break;
            case GGUF_TYPE_FLOAT32:
                meta_f32[key] = *(float*)p; p += 4; break;
            case GGUF_TYPE_BOOL:
                meta_u32[key] = *(uint8_t*)p; p += 1; break;
            case GGUF_TYPE_UINT8:
                meta_u32[key] = *(uint8_t*)p; p += 1; break;
            case GGUF_TYPE_INT8:
                meta_u32[key] = *(uint8_t*)p; p += 1; break;
            case GGUF_TYPE_UINT16:
                meta_u32[key] = *(uint16_t*)p; p += 2; break;
            case GGUF_TYPE_INT16:
                meta_u32[key] = *(uint16_t*)p; p += 2; break;
            case GGUF_TYPE_FLOAT64:
                meta_f32[key] = (float)*(double*)p; p += 8; break;
            case GGUF_TYPE_INT64:
                meta_u64[key] = *(uint64_t*)p; p += 8; break;
            case GGUF_TYPE_ARRAY: {
                uint32_t atype = *(uint32_t*)p; p += 4;
                uint64_t count = *(uint64_t*)p; p += 8;
                if (atype == GGUF_TYPE_STRING) {
                    std::vector<std::string> arr;
                    for (uint64_t j = 0; j < count; j++) {
                        uint64_t slen = *(uint64_t*)p; p += 8;
                        arr.push_back(std::string((char*)p, slen)); p += slen;
                    }
                    meta_str_arr[key] = arr;
                } else if (atype == GGUF_TYPE_INT32 || atype == GGUF_TYPE_UINT32) {
                    std::vector<int32_t> arr(count);
                    for (uint64_t j = 0; j < count; j++) { arr[j] = *(int32_t*)p; p += 4; }
                    meta_i32_arr[key] = arr;
                } else if (atype == GGUF_TYPE_BOOL || atype == GGUF_TYPE_UINT8 || atype == GGUF_TYPE_INT8) {
                    std::vector<uint8_t> arr(count);
                    for (uint64_t j = 0; j < count; j++) { arr[j] = *(uint8_t*)p; p += 1; }
                    meta_bool_arr[key] = arr;
                } else if (atype == GGUF_TYPE_FLOAT32) {
                    std::vector<float> arr(count);
                    for (uint64_t j = 0; j < count; j++) { arr[j] = *(float*)p; p += 4; }
                    meta_f32_arr[key] = arr;
                } else {
                    // Skip other array types
                    static const size_t type_sizes[] = {1,1,2,2,4,4,4,1,0,0,8,8,8};
                    if (atype < 13 && atype != GGUF_TYPE_STRING && atype != GGUF_TYPE_ARRAY) {
                        p += count * type_sizes[atype];
                    }
                }
                break;
            }
        }
        return p;
    }
};

// 양자화 블록 크기
inline uint64_t TensorInfo::byte_size() const {
    uint64_t ne = num_elements();
    switch (type) {
        case GGML_TYPE_F32:  return ne * 4;
        case GGML_TYPE_F16:  return ne * 2;
        case GGML_TYPE_BF16: return ne * 2;
        case GGML_TYPE_Q4_0: return ne / 32 * 18;
        case GGML_TYPE_Q4_1: return ne / 32 * 20;
        case GGML_TYPE_Q5_0: return ne / 32 * 22;
        case GGML_TYPE_Q5_1: return ne / 32 * 24;
        case GGML_TYPE_Q8_0: return ne / 32 * 34;
        case GGML_TYPE_Q2_K: return ne / 256 * 84;
        case GGML_TYPE_Q3_K: return ne / 256 * 110;
        case GGML_TYPE_Q4_K: return ne / 256 * 144;
        case GGML_TYPE_Q5_K: return ne / 256 * 176;
        case GGML_TYPE_Q6_K: return ne / 256 * 210;
        case GGML_TYPE_Q8_K: return ne / 256 * 292;
        default: return ne * 2;  // assume fp16
    }
}

// Bytes for one row (single dim along row_dim) of a quantized tensor.
// Used by gemma MoE expert tensor row stride math.
inline size_t ggml_row_bytes(ggml_type type, int row_dim) {
    switch (type) {
        case GGML_TYPE_F32:  return (size_t)row_dim * 4;
        case GGML_TYPE_F16:  return (size_t)row_dim * 2;
        case GGML_TYPE_BF16: return (size_t)row_dim * 2;
        case GGML_TYPE_Q4_0: return (size_t)row_dim / 32 * 18;
        case GGML_TYPE_Q4_1: return (size_t)row_dim / 32 * 20;
        case GGML_TYPE_Q5_0: return (size_t)row_dim / 32 * 22;
        case GGML_TYPE_Q5_1: return (size_t)row_dim / 32 * 24;
        case GGML_TYPE_Q8_0: return (size_t)row_dim / 32 * 34;
        case GGML_TYPE_Q2_K: return (size_t)row_dim / 256 * 84;
        case GGML_TYPE_Q3_K: return (size_t)row_dim / 256 * 110;
        case GGML_TYPE_Q4_K: return (size_t)row_dim / 256 * 144;
        case GGML_TYPE_Q5_K: return (size_t)row_dim / 256 * 176;
        case GGML_TYPE_Q6_K: return (size_t)row_dim / 256 * 210;
        case GGML_TYPE_Q8_K: return (size_t)row_dim / 256 * 292;
        default: return (size_t)row_dim * 2;
    }
}
