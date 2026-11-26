/***************************************************************************************************
 * Copyright (c) 2024 - 2025 Moore Threads Technology Co., Ltd("Moore Threads"). All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
#pragma once

#include <mute/config.hpp>

#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ == 220) && !defined(MUTE_ARCH_MMA_MP22_ENABLED)
#  define MUTE_ARCH_MMA_MP22_ENABLED
#endif

#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ >= 310)
#  if !defined(MUTE_ARCH_SIMD_MATH_ENABLED)
#    define MUTE_ARCH_SIMD_MATH_ENABLED
#  endif
#endif

#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ == 310)
#  if !defined(MUTE_ARCH_MMA_MP31_ENABLED)
#    define MUTE_ARCH_MMA_MP31_ENABLED
#  endif
#  if !defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
#    define MUTE_ARCH_SQMMA_MP31_ENABLED
#  endif
#endif


#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ == 320)
#  if !defined(MUTE_ARCH_SQMMA_MP32_ENABLED)
#    define MUTE_ARCH_SQMMA_MP32_ENABLED
#  endif
#  if !defined(MUTE_ARCH_TME_MP32_ACTIVATED)
#    define MUTE_ARCH_TME_MP32_ACTIVATED
#  endif
#endif

#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ >= 310)
#  if !defined(MUTE_ARCH_ROBUST_BUFFER_ACCESS_ACTIVATED)
#    define MUTE_ARCH_ROBUST_BUFFER_ACCESS_ACTIVATED
#  endif
#  if !defined(MUTE_ARCH_TME_MP31_ACTIVATED)
#    define MUTE_ARCH_TME_MP31_ACTIVATED
#  endif
#  if !defined(MUTE_ARCH_ASYNC_BARRIER_ENABLED)
#    define MUTE_ARCH_ASYNC_BARRIER_ENABLED
#  endif
#endif

#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED) || defined(MUTE_ARCH_SQMMA_MP32_ENABLED)
#  if !defined(MUTE_ARCH_SQMMA_ENABLED)
#    define MUTE_ARCH_SQMMA_ENABLED
#  endif
#endif
