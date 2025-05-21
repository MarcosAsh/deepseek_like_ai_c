#include "memory_pool.hpp"
#include <cstdlib>
#include <stdexcept>

// Retrieve singleton instance
UnifiedMemoryManager& UnifiedMemoryManager::instance() {
    static UnifiedMemoryManager inst;
    return inst;
}

UnifiedMemoryManager::UnifiedMemoryManager() = default;

UnifiedMemoryManager::~UnifiedMemoryManager() {
    std::lock_guard<std::mutex> lock(mu_);
    if (pool_) {
        std::free(pool_);
        pool_ = nullptr;
    }
    // Free any remaining fallback allocations (if any)
    for (auto& kv : allocations_) {
        void* ptr = kv.first;
        if (!(ptr >= pool_ && ptr < pool_ + max_on_chip_)) {
            std::free(ptr);
        }
    }
    allocations_.clear();
}

void UnifiedMemoryManager::init(std::size_t max_on_chip_bytes) {
    std::lock_guard<std::mutex> lock(mu_);
    if (pool_) {
        // Already initialized
        return;
    }
    pool_ = static_cast<char*>(std::malloc(max_on_chip_bytes));
    if (!pool_) throw std::bad_alloc();
    max_on_chip_ = max_on_chip_bytes;
    allocated_on_chip_ = 0;
}

void* UnifiedMemoryManager::allocate(std::size_t bytes) {
    std::lock_guard<std::mutex> lock(mu_);
    if (pool_ && allocated_on_chip_ + bytes <= max_on_chip_) {
        void* ptr = pool_ + allocated_on_chip_;
        allocations_[ptr] = bytes;
        allocated_on_chip_ += bytes;
        return ptr;
    } else {
        void* ptr = std::malloc(bytes);
        if (!ptr) throw std::bad_alloc();
        allocations_[ptr] = bytes;
        return ptr;
    }
}

void UnifiedMemoryManager::deallocate(void* ptr, std::size_t /*bytes*/) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = allocations_.find(ptr);
    if (it != allocations_.end()) {
        bool is_pool = pool_ && (ptr >= pool_ && ptr < pool_ + max_on_chip_);
        if (is_pool) {
            allocated_on_chip_ -= it->second;
        } else {
            std::free(ptr);
        }
        allocations_.erase(it);
    } else {
        // Unknown pointer, free for safety
        std::free(ptr);
    }
}