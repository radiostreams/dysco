// K5 — apply the normalization factors.
//
// K5a divides every value by its channel RMS (after K2 has computed all of
// them, exactly like the CPU does in a second pass), K5b multiplies by
// g_a1 * g_a2 for the polarization currently being processed.  The operation
// order (divide first, then multiply) is kept identical to the CPU encoder so
// that the double results match bit for bit.
//
// K5b processes all polarizations in a single launch.  The per-polarization
// antenna RMS factors (K4, [pol * nAnt + ant]) must already be computed; the
// orchestrator therefore runs K3/K4 for every polarization first.  This is
// bit-identical to the CPU order: `calculateAntennaeRMS(data, p)` and the
// pol-p antenna normalization only ever touch the pol-p elements, so the
// CPU's interleaved per-pol loop carries no data dependency.  The single
// launch also makes every thread read nPol *contiguous* doubles per row,
// which avoids the 4x DRAM sector amplification of the strided per-pol
// layout (each of the four per-pol launches otherwise touches only 8 of
// every 32 bytes of a sector).

#include "cuda_kernels.h"

namespace dyscostman {
namespace gpu {

namespace {

__global__ void applyChannelNormKernel(cuda::std::span<double> re,
                                       cuda::std::span<double> im,
                                       std::size_t visPerRow,
                                       cuda::std::span<const float> channelRms) {
  const std::size_t nVis = re.size();
  const std::size_t stride = gridStrideSize();
  for (std::size_t i = threadFlattenedId(); i < nVis; i += stride) {
    const double rms = static_cast<double>(channelRms[i % visPerRow]);
    // Match the CPU encoder: a zero RMS leaves the (zero) values untouched
    // instead of dividing 0/0 and producing NaNs.
    if (rms != 0.0) {
      re[i] /= rms;
      im[i] /= rms;
    }
  }
}

__global__ void applyAntennaNormKernel(
    MdSpan2D<double> re, MdSpan2D<double> im,
    cuda::std::span<const std::int32_t> ant1,
    cuda::std::span<const std::int32_t> ant2, std::size_t nChannels,
    std::size_t nPol, std::size_t nAnt,
    cuda::std::span<const double> rmsPerAntenna) {
  // Grid: x = channel tile within a row, y = row.  Adjacent threads read
  // adjacent channels, so the re/im accesses are contiguous.
  const std::size_t row = blockIdx.y;
  if (row >= re.extent(0)) return;
  const std::size_t ch = blockIdx.x * blockDim.x + threadIdx.x;
  if (ch >= nChannels) return;
  const std::size_t a1 = static_cast<std::size_t>(ant1[row]);
  const std::size_t a2 = static_cast<std::size_t>(ant2[row]);
  for (std::size_t pol = 0; pol != nPol; ++pol) {
    const double mul = rmsPerAntenna[pol * nAnt + a1] *
                       rmsPerAntenna[pol * nAnt + a2];
    // Match the CPU encoder: a zero gain product leaves the (zero) values
    // untouched instead of multiplying by 1/0 = inf.
    const double fac = (mul == 0.0) ? 0.0 : 1.0 / mul;
    const std::size_t vi = ch * nPol + pol;
    re(row, vi) *= fac;
    im(row, vi) *= fac;
  }
}

}  // namespace

void k5a_apply_channel_norm(cuda::std::span<double> re,
                            cuda::std::span<double> im, std::size_t visPerRow,
                            cuda::std::span<const float> channelRms,
                            dysco_stream_t stream) {
  constexpr std::uint32_t kThreads = kThreadBlockSize;
  const std::size_t nVis = re.size();
  const auto blocks = gridFor(nVis, kThreads);
  DYSCO_LAUNCH_1D(applyChannelNormKernel, blocks, kThreads, stream, re, im,
                  visPerRow, channelRms);
}

void k5b_apply_antenna_norm(MdSpan2D<double> re, MdSpan2D<double> im,
                            cuda::std::span<const std::int32_t> ant1,
                            cuda::std::span<const std::int32_t> ant2,
                            std::size_t nChannels, std::size_t nPol,
                            std::size_t nAnt,
                            cuda::std::span<const double> rmsPerAntenna,
                            dysco_stream_t stream) {
  // 1024 threads per block: one block covers the widest channel count seen
  // in practice (and measures fastest on both the SKA and MWA datasets).
  constexpr std::uint32_t kThreads = 1024;
  const dim3 grid(gridFor(nChannels, kThreads),
                  static_cast<std::uint32_t>(re.extent(0)));
  applyAntennaNormKernel<<<grid, kThreads, 0, stream>>>(re, im, ant1, ant2,
                                                        nChannels, nPol, nAnt,
                                                        rmsPerAntenna);
}

}  // namespace gpu
}  // namespace dyscostman
