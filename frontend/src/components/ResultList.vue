<script setup lang="ts">
import { Clock3, Play, Target } from 'lucide-vue-next'
import type { Moment } from '../types/search'

defineProps<{
  moments: Moment[]
  activeMoment: number | null
}>()

const emit = defineEmits<{
  select: [index: number]
}>()

function formatTime(seconds: number) {
  const minutes = Math.floor(seconds / 60)
  const remainder = Math.floor(seconds % 60)
  return `${String(minutes).padStart(2, '0')}:${String(remainder).padStart(2, '0')}`
}
</script>

<template>
  <div class="results-list">
    <button
      v-for="(moment, index) in moments"
      :key="`${moment.start}-${moment.end}`"
      class="result-row"
      :class="{ active: activeMoment === index }"
      @click="emit('select', index)"
    >
      <span class="rank">{{ String(index + 1).padStart(2, '0') }}</span>
      <span class="result-main">
        <span class="result-time">
          {{ formatTime(moment.start) }} — {{ formatTime(moment.end) }}
        </span>
        <span class="result-meta">
          <Clock3 :size="13" /> {{ (moment.end - moment.start).toFixed(1) }} 秒
          <Target :size="13" /> 排序分数 {{ moment.score.toFixed(3) }}
        </span>
      </span>
      <span class="score-bar" aria-hidden="true">
        <i :style="{ width: `${Math.min(100, Math.max(0, moment.score * 100))}%` }"></i>
      </span>
      <span class="play-result"><Play :size="16" fill="currentColor" /></span>
    </button>
  </div>
</template>
