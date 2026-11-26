/***************************************************************************************************
 * Copyright (c) 2024 - 2026 Moore Threads Technology Co., Ltd("Moore Threads"). All rights reserved.
 * Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
#pragma once

#include <mute/config.hpp>           // MUTE_HOST_DEVICE

#include <mute/layout.hpp>
#include <mute/tensor_impl.hpp>
#include <mute/pointer_flagged.hpp>

namespace mute
{

////////////////////////////////
// Layout 2D to Console table //
////////////////////////////////

template <class Layout>
MUTE_HOST_DEVICE
void
print_layout(Layout const& layout)  // (m,n) -> idx
{
  MUTE_STATIC_ASSERT_V(rank(layout) == Int<2>{});

  int idx_width = num_digits(cosize(layout)) + 2;

  print(layout); print("\n");

  // Column indices
  print("    ");
  for (int n = 0; n < size<1>(layout); ++n) { printf("  %*d ", idx_width-2, n); }
  printf("\n");

  // Print out A m-by-n
  for (int m = 0; m < size<0>(layout); ++m) {
    // Header
    print("    ");
    for (int n = 0; n < size<1>(layout); ++n) {
      printf("+");
      for (int i = 0; i < idx_width; ++i) {
        printf("-");
      }
    }
    printf("+\n");
    // Values
    printf("%2d  ", m);  // Row indices
    for (int n = 0; n < size<1>(layout); ++n) { printf("| %*d ", idx_width-2, int(layout(m,n))); }
    printf("|\n");
  }
  // Footer
  print("    ");
  for (int n = 0; n < size<1>(layout); ++n) {
    printf("+");
    for (int i = 0; i < idx_width; ++i) {
      printf("-");
    }
  }
  printf("+\n");
}

// Capture and cast smem_ptr_flag Layouts to offset-0 layouts
template <class SwizzleFn, int B, class Layout>
MUTE_HOST_DEVICE
void
print_layout(ComposedLayout<SwizzleFn,smem_ptr_flag_bits<B>,Layout> const& layout)
{
  print_layout(as_position_independent_swizzle_layout(layout));
}

////////////////////////////////
// Tensor 1D,2D,3D,4D Console //
////////////////////////////////

template <class Engine, class Layout>
MUTE_HOST_DEVICE
void
print_tensor(Tensor<Engine,Layout> const& tensor, bool print_type = true)
{
  if (print_type) {
    print(tensor); print(":\n");
  }

  if constexpr (Layout::rank == 1)
  {
    for (int m = 0; m < size(tensor); ++m) {
      pretty_print(tensor(m));
      printf("\n");
    }
  } else
  if constexpr (Layout::rank == 2)
  {
    for (int m = 0; m < size<0>(tensor); ++m) {
      for (int n = 0; n < size<1>(tensor); ++n) {
        pretty_print(tensor(m,n));
      }
      printf("\n");
    }
  } else
  if constexpr (Layout::rank == 3)
  {
    print_tensor(tensor(_,_,0), false);
    for (int k = 1; k < size<2>(tensor); ++k) {
      for (int i = 0; i < 5*size<1>(tensor); ++i) { print("-"); } print("\n");
      print_tensor(tensor(_,_,k), false);
    }
  } else
  if constexpr (Layout::rank == 4)
  {
    print_tensor(tensor(_,_,_,0), false);
    for (int p = 1; p < size<3>(tensor); ++p) {
      for (int i = 0; i < 5*size<1>(tensor); ++i) { print("="); } print("\n");
      print_tensor(tensor(_,_,_,p), false);
    }
  }
}

#if !defined(__MUSACC_RTC__)
template <class Engine, class Layout>
MUTE_HOST
std::ostream&
print_tensor_os(std::ostream& os, Tensor<Engine,Layout> const& tensor)
{
  int digits = 9;

  if constexpr (Layout::rank == 1)
  {
    for (int m = 0; m < size(tensor); ++m) {
      os << std::setw(digits) << tensor(m) << std::endl;
    }
  } else
  if constexpr (Layout::rank == 2)
  {
    for (int m = 0; m < size<0>(tensor); ++m) {
      for (int n = 0; n < size<1>(tensor); ++n) {
        os << std::setw(digits) << tensor(m,n);
      }
      os << std::endl;
    }
  } else
  if constexpr (Layout::rank == 3)
  {
    print_tensor_os(os, tensor(_,_,0));
    for (int k = 1; k < size<2>(tensor); ++k) {
      for (int i = 0; i < digits*size<1>(tensor); ++i) { os << "-"; } os << std::endl;
      print_tensor_os(os, tensor(_,_,k));
    }
  } else
  if constexpr (Layout::rank == 4)
  {
    print_tensor_os(os, tensor(_,_,_,0));
    for (int p = 1; p < size<3>(tensor); ++p) {
      for (int i = 0; i < digits*size<1>(tensor); ++i) { os << "="; } os << std::endl;
      print_tensor_os(os, tensor(_,_,_,p));
    }
  }

  return os;
}

template <class Engine, class Layout>
MUTE_HOST
std::ostream&
operator<<(std::ostream& os, Tensor<Engine,Layout> const& tensor)
{
  os << tensor.layout() << std::endl;
  return print_tensor_os(os, tensor);
}
#endif // !defined(__MUSACC_RTC__)

} // end namespace mute
