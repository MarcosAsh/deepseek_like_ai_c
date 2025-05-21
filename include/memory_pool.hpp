// Unified memory manager for on-chip (unified) memory pools
#pragma once
#include <cstddef>
#include <mutex>
#include <unordered_map>

class UnifiedMemoryManager {
public:
    // Retrieve the singleton instance
    static UnifiedMemoryManager& instance();

    // Initialize the on-chip pool with given size in bytes (call once)
    void init(std::size_t max_on_chip_bytes);

    // Allocate a block of memory of given size
    // Returns pointer to on-chip pool if available, otherwise fallback to heap
    void* allocate(std::size_t bytes);

    // Deallocate a previously allocated block
    void deallocate(void* ptr, std::size_t bytes);

    UnifiedMemoryManager(const UnifiedMemoryManager&) = delete;
    UnifiedMemoryManager& operator=(const UnifiedMemoryManager&) = delete;

private:
    UnifiedMemoryManager();
    ~UnifiedMemoryManager();

    std::mutex mu_;
    std::size_t max_on_chip_ = 0;
    std::size_t allocated_on_chip_ = 0;
    char* pool_ = nullptr;
    std::unordered_map<void*, std::size_t> allocations_;
};