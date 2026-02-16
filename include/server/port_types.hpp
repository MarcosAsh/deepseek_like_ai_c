#pragma once
#include <string>
#include <vector>
#include <variant>
#include <memory>
#include "tensor.hpp"
#include "autodiff.hpp"

namespace server {

enum class PortType {
    TEXT,
    TOKEN_IDS,
    TENSOR,
    AD_TENSOR,
    SCALAR,
    INT
};

inline std::string port_type_name(PortType t) {
    switch (t) {
        case PortType::TEXT:       return "TEXT";
        case PortType::TOKEN_IDS:  return "TOKEN_IDS";
        case PortType::TENSOR:     return "TENSOR";
        case PortType::AD_TENSOR:  return "AD_TENSOR";
        case PortType::SCALAR:     return "SCALAR";
        case PortType::INT:        return "INT";
    }
    return "UNKNOWN";
}

inline PortType port_type_from_string(const std::string& s) {
    if (s == "TEXT")       return PortType::TEXT;
    if (s == "TOKEN_IDS")  return PortType::TOKEN_IDS;
    if (s == "TENSOR")     return PortType::TENSOR;
    if (s == "AD_TENSOR")  return PortType::AD_TENSOR;
    if (s == "SCALAR")     return PortType::SCALAR;
    if (s == "INT")        return PortType::INT;
    throw std::runtime_error("Unknown port type: " + s);
}

using PortValue = std::variant<
    std::string,                    // TEXT
    std::vector<int>,               // TOKEN_IDS
    Tensor,                         // TENSOR
    std::shared_ptr<ADTensor>,      // AD_TENSOR
    float,                          // SCALAR
    int                             // INT
>;

struct PortDescriptor {
    std::string name;
    PortType type;
    bool optional = false;
};

} // namespace server
