<script setup lang="ts">
import { computed } from 'vue'
import type { Moment } from '../types/search'

const props = defineProps<{
  duration: number
  currentTime: number
  moments: Moment[]
  saliencyScores: number[]
  clipDuration: number
  activeMoment: number | null
}>()

const emit = defineEmits<{
  seek: [time: number]
  selectMoment: [index: number]
}>()

const normalizedScores = computed(() => {
  if (!props.saliencyScores.length) return []
  const min = Math.min(...props.saliencyScores)
  const max = Math.max(...props.saliencyScores)
  const range = Math.max(max - min, 1e-8)
  return props.saliencyScores.map((score) => (score - min) / range)
})

const progress = computed(() =>
  props.duration > 0 ? Math.min(100, (props.currentTime / props.duration) * 100) : 0,
)

function seekFromPointer(event: MouseEvent) {
  if (!props.duration) return
  const target = event.currentTarget as HTMLElement
  const bounds = target.getBoundingClientRect()
  const ratio = Math.min(1, Math.max(0, (event.clientX - bounds.left) / bounds.width))
  emit('seek', ratio * props.duration)
}

function percent(value: number) {
  return props.duration > 0 ? `${Math.min(100, Math.max(0, (value / props.duration) * 100))}%` : '0%'
}
</script>

<template>
  <section class="timeline-panel" aria-label="视频分析时间轴">
    <div class="timeline-heading">
      <div>
        <span class="eyebrow">TEMPORAL MAP</span>
        <h2>语义响应时间轴</h2>
      </div>
      <div class="timeline-legend">
        <span><i class="legend-swatch moment-swatch"></i>检索区间</span>
        <span><i class="legend-swatch heat-swatch"></i>高光强度</span>
      </div>
    </div>

    <div class="timeline-label">检索结果</div>
    <div class="moment-track" @click="seekFromPointer">
      <button
        v-for="(moment, index) in moments"
        :key="`${moment.start}-${moment.end}`"
        class="moment-segment"
        :class="{ active: activeMoment === index }"
        :style="{ left: percent(moment.start), width: percent(moment.end - moment.start) }"
        :title="`结果 ${index + 1}: ${moment.start}s – ${moment.end}s`"
        :aria-label="`跳转到结果 ${index + 1}`"
        @click.stop="emit('selectMoment', index)"
      >
        <span>{{ index + 1 }}</span>
      </button>
      <div class="playhead" :style="{ left: `${progress}%` }"></div>
    </div>

    <div class="timeline-label heat-label">高光评分</div>
    <div class="heat-track" @click="seekFromPointer">
      <div
        v-for="(score, index) in normalizedScores"
        :key="index"
        class="heat-cell"
        :style="{
          backgroundColor: `rgba(235, 91, 59, ${0.1 + score * 0.82})`,
          width: percent(Math.max(0, Math.min(clipDuration, duration - index * clipDuration))),
        }"
        :title="`${index * clipDuration}s – ${Math.min((index + 1) * clipDuration, duration)}s · 强度 ${score.toFixed(2)}`"
      ></div>
      <div class="playhead heat-playhead" :style="{ left: `${progress}%` }"></div>
    </div>

    <div class="time-scale">
      <span>00:00</span>
      <span>{{ Math.round(duration / 2) }}s</span>
      <span>{{ Math.round(duration) }}s</span>
    </div>
  </section>
</template>
