// npy.h — just enough of the .npy format to read the float32 reference arrays in tests/refs.
#pragma once

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace st {

inline std::vector<float> read_npy_f32(const std::string& path, std::vector<size_t>& shape) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + path);
    char magic[8];
    if (fread(magic, 1, 8, f) != 8 || std::string(magic + 1, 5) != "NUMPY") {
        fclose(f);
        throw std::runtime_error(path + ": not a .npy file");
    }
    uint32_t hlen = 0;
    if (magic[6] == 1) {
        uint16_t h = 0;
        if (fread(&h, 2, 1, f) != 1) { fclose(f); throw std::runtime_error(path + ": truncated"); }
        hlen = h;
    } else if (fread(&hlen, 4, 1, f) != 1) {
        fclose(f);
        throw std::runtime_error(path + ": truncated");
    }
    std::string hdr(hlen, '\0');
    if (fread(&hdr[0], 1, hlen, f) != hlen) { fclose(f); throw std::runtime_error(path + ": truncated"); }
    if (hdr.find("'<f4'") == std::string::npos || hdr.find("'fortran_order': False") == std::string::npos) {
        fclose(f);
        throw std::runtime_error(path + ": expected little-endian float32, C order");
    }
    shape.clear();
    const size_t a = hdr.find('(', hdr.find("'shape'")), b = hdr.find(')', a);
    size_t n = 1;
    for (size_t i = a + 1; i < b;) {
        while (i < b && (hdr[i] == ' ' || hdr[i] == ',')) i++;
        if (i >= b) break;
        size_t j = i;
        while (j < b && hdr[j] >= '0' && hdr[j] <= '9') j++;
        shape.push_back(std::stoull(hdr.substr(i, j - i)));
        n *= shape.back();
        i = j;
    }
    std::vector<float> data(n);
    const size_t got = fread(data.data(), sizeof(float), n, f);
    fclose(f);
    if (got != n) throw std::runtime_error(path + ": truncated data");
    return data;
}

} // namespace st
