#include "parse_ppm.h"
#include <fstream>
#include <iostream>

using namespace std;

optional<tuple<PpmHeader, vector<PpmPixel>>> parse_ppm(const std::string &image_path) {
    ifstream file(image_path);

    if (!file.is_open()) {
        cerr << "Failed to open file: " << image_path << endl;
        return nullopt;
    }

    PpmHeader header;
    string magic;
    if(!(file >> magic)) {
        cerr << "Failed to read header" << endl;
        return nullopt;
    }
    if(magic != "P3"){
        cerr << "Unsuported PPM version: " << magic << endl;
        cerr << "Use version P3" << endl;
        return nullopt;
    }

    // Skip newline
    file.ignore(1);
    // Skip comments
    while (file.peek() == '#') {
        file.ignore(1<<30, '\n');
    }

    if(!(file >> header.width >> header.height >> header.max_value)) {
        cerr << "Failed to read header" << endl;
        return nullopt;
    }

    vector<PpmPixel> pixels;
    for (int i = 0; i < header.width * header.height; i++) {
        int r,g,b;
        if(!(file >> r >> g >> b)) {
            cerr << "Failed to read pixel: " << i << endl;
            return nullopt;
        }
        if(max(max(r,g),b) > header.max_value) {
            cerr << "Pixel value out of range: " << r << " " << g << " " << b << endl;
            return nullopt;
        }
        pixels.push_back(PpmPixel(r,g,b));
    }

    return optional(make_tuple(header, pixels));
}
