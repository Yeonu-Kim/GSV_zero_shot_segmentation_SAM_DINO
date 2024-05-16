#include <cuda_runtime_api.h>

namespace groundingdinobuild {
int get_cudart_version() {
  return CUDART_VERSION;
}
} // namespace groundingdino
