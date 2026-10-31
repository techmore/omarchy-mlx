// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Fence on top of the event implementation. The events are real timeline
// semaphores on GPU streams, so fences carry real device ordering.

#include "mlx/fence.h"

#include "mlx/event.h"

#include <mutex>

namespace mlx::core {

struct FenceImpl {
  uint32_t count;
  Event event;
  std::mutex mutex;

  FenceImpl(uint32_t count, Stream s) : count(count), event(s) {}
};

Fence::Fence(Stream s) {
  fence_ = std::make_shared<FenceImpl>(0, s);
}

void Fence::wait(Stream s, const array&, uint32_t value) {
  if (value == 0) {
    return;
  }
  auto& f = cast<FenceImpl>();
  Event event;
  {
    std::lock_guard<std::mutex> lock(f.mutex);
    // Event copies share the underlying event but have independent values.
    event = f.event;
  }
  event.set_value(value);
  event.wait(s);
}

uint32_t Fence::update(Stream s, const array&, bool) {
  auto& f = cast<FenceImpl>();
  std::lock_guard<std::mutex> lock(f.mutex);
  if (std::getenv("MLX_OMARCHY_TRACE_DISPATCH") != nullptr) {
    fprintf(stderr, "[rtmod] FENCE-UPDATE count=%u st=%d\n", f.count, s.index);
  }
  f.count++;
  f.event.set_value(f.count);
  f.event.signal(s);
  return f.count;
}

} // namespace mlx::core
