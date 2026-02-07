#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <string>

class Tensor {
public:
    Tensor() = default;
    Tensor(std::vector<float> data, std::vector<int> shape);

    Tensor(const std::string& filename);

    const std::vector<float>& data() const;
    std::vector<float>& data();

    const std::vector<int>& shape() const;

    int ndim() const;
    int size() const;

    void reshape(const std::vector<int>& new_shape);
    Tensor transpose() const;

    void print() const;

private:
    std::vector<float> data_;
    std::vector<int> shape_;
};

#endif

