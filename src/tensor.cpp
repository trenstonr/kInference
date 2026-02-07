#include "../include/tensor.hpp"
#include <iostream>
#include <fstream>
#include <stdexcept>

Tensor::Tensor(std::vector<float> data, std::vector<int> shape)
    : data_(std::move(data)), shape_(std::move(shape)) {}

Tensor::Tensor(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file)
        throw std::runtime_error("Cannot open file: " + filename);
    
    int ndim;
    file.read(reinterpret_cast<char*>(&ndim), sizeof(int));
    
    shape_.resize(ndim);
    file.read(reinterpret_cast<char*>(shape_.data()), ndim * sizeof(int));
    
    int total_size = 1;
    for (int dim : shape_) {
        total_size *= dim;
    }
    
    data_.resize(total_size);
    file.read(reinterpret_cast<char*>(data_.data()), total_size * sizeof(float));
    
    file.close();
}

const std::vector<float>& Tensor::data() const { return data_; }

std::vector<float>& Tensor::data() { return data_; }

const std::vector<int>& Tensor::shape() const { return shape_; }

int Tensor::ndim() const { return shape_.size(); }

int Tensor::size() const { return data_.size(); }

void Tensor::reshape(const std::vector<int>& new_shape) {
	int prod = 1;
    for (int d : new_shape) prod *= d;
    if (prod != size())
        throw std::runtime_error("reshape: size mismatch");
    shape_ = new_shape;
}

Tensor Tensor::transpose() const {
    if (shape_.size() != 2)
        throw std::runtime_error("transpose requires 2D tensor");

    int rows = shape_[0];
    int cols = shape_[1];

    std::vector<float> out(data_.size());
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            out[c * rows + r] = data_[r * cols + c];

    return Tensor(out, {cols, rows});
}

void Tensor::print() const {
    if (shape_.size() != 2) {
        std::cout << "[ tensor ndim=" << ndim() << " size=" << size() << " ]\n";
        return;
    }

    int rows = shape_[0];
    int cols = shape_[1];

    std::cout << "[\n";
    for (int r = 0; r < rows; ++r) {
        std::cout << "  [ ";
        for (int c = 0; c < cols; ++c) {
            std::cout << data_[r * cols + c];
            if (c + 1 < cols) std::cout << ", ";
        }
        std::cout << " ]\n";
    }
    std::cout << "]\n";
}

