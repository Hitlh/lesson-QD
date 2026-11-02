<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import {
  AlertCircle,
  Check,
  ChevronDown,
  FileVideo2,
  LoaderCircle,
  Play,
  Search,
  Sparkles,
  Upload,
  Video,
} from 'lucide-vue-next'
import ResultList from './components/ResultList.vue'
import VideoTimeline from './components/VideoTimeline.vue'
import { ApiError, getBackendHealth, searchVideo, uploadVideo } from './services/search'
import type { SearchResponse } from './types/search'

type Status = 'idle' | 'ready' | 'uploading' | 'searching' | 'success' | 'error'
type BackendStatus = 'checking' | 'online' | 'offline'

const fileInput = ref<HTMLInputElement | null>(null)
const videoElement = ref<HTMLVideoElement | null>(null)
const videoUrl = ref('')
const selectedFile = ref<File | null>(null)
const uploadedVideoId = ref('')
const videoName = ref('')
const videoSize = ref(0)
const duration = ref(0)
const currentTime = ref(0)
const query = ref('A person prepares and serves food.')
const topK = ref(3)
const status = ref<Status>('idle')
const errorMessage = ref('')
const result = ref<SearchResponse | null>(null)
const activeMoment = ref<number | null>(null)
const stopAt = ref<number | null>(null)
const backendStatus = ref<BackendStatus>('checking')
const backendDevice = ref('')
const backendModelLabel = ref('CLIP')
const backendMaxDuration = ref<number | null>(null)
const backendMaxUploadMb = ref<number | null>(null)

const isBusy = computed(() => status.value === 'uploading' || status.value === 'searching')
const durationTooLong = computed(
  () => backendMaxDuration.value !== null && duration.value > backendMaxDuration.value + 0.05,
)
const fileTooLarge = computed(
  () => backendMaxUploadMb.value !== null && videoSize.value > backendMaxUploadMb.value * 1024 * 1024,
)
const canSearch = computed(
  () => Boolean(selectedFile.value && Number.isFinite(duration.value) && duration.value > 0 && query.value.trim())
    && !durationTooLong.value && !fileTooLarge.value && !isBusy.value,
)
const fileSizeLabel = computed(() => `${(videoSize.value / 1024 / 1024).toFixed(1)} MB`)
const backendLabel = computed(() => {
  if (backendStatus.value === 'checking') return '正在检查模型服务'
  if (backendStatus.value === 'online') return `模型服务可用${backendDevice.value ? ` · ${backendDevice.value}` : ''}`
  return '模型服务未连接'
})

function openFilePicker() {
  fileInput.value?.click()
}

function selectFile(event: Event) {
  if (isBusy.value) return
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]
  if (!file) return
  if (!/\.(mp4|webm|mov)$/i.test(file.name)) {
    status.value = 'error'
    errorMessage.value = '请选择 MP4、WebM 或 MOV 视频文件'
    return
  }

  if (videoUrl.value) URL.revokeObjectURL(videoUrl.value)
  selectedFile.value = file
  uploadedVideoId.value = ''
  videoUrl.value = URL.createObjectURL(file)
  videoName.value = file.name
  videoSize.value = file.size
  duration.value = 0
  currentTime.value = 0
  result.value = null
  activeMoment.value = null
  stopAt.value = null
  status.value = 'ready'
  errorMessage.value = ''
}

function onMetadataLoaded() {
  duration.value = videoElement.value?.duration || 0
}

function onTimeUpdate() {
  const player = videoElement.value
  if (!player) return
  currentTime.value = player.currentTime
  if (stopAt.value !== null && player.currentTime >= stopAt.value) {
    player.pause()
    stopAt.value = null
  }
}

async function runSearch() {
  if (!canSearch.value || !selectedFile.value) return
  result.value = null
  activeMoment.value = null
  errorMessage.value = ''
  stopAt.value = null
  videoElement.value?.pause()

  const file = selectedFile.value
  const requestQuery = query.value.trim()
  const requestTopK = topK.value
  async function uploadSelectedVideo() {
    status.value = 'uploading'
    const uploaded = await uploadVideo(file, duration.value)
    uploadedVideoId.value = uploaded.video_id
    duration.value = uploaded.duration
  }
  async function searchUploadedVideo() {
    status.value = 'searching'
    return searchVideo({
      video_id: uploadedVideoId.value,
      query: requestQuery,
      top_k: requestTopK,
    })
  }

  try {
    if (!uploadedVideoId.value) await uploadSelectedVideo()
    try {
      result.value = await searchUploadedVideo()
    } catch (error) {
      if (!(error instanceof ApiError) || error.status !== 404) throw error
      // A backend restart invalidates its in-memory video IDs. Retry once with
      // a fresh upload, while keeping the same selected file and query.
      uploadedVideoId.value = ''
      await uploadSelectedVideo()
      result.value = await searchUploadedVideo()
    }
    backendStatus.value = 'online'
    status.value = 'success'
  } catch (error) {
    if (error instanceof TypeError) backendStatus.value = 'offline'
    status.value = 'error'
    errorMessage.value = error instanceof Error ? error.message : '检索失败，请稍后重试'
  }
}

function seekTo(time: number) {
  if (!videoElement.value) return
  stopAt.value = null
  const target = Math.min(duration.value, Math.max(0, time))
  videoElement.value.currentTime = target
  currentTime.value = target
}

function playMoment(index: number) {
  const moment = result.value?.moments[index]
  if (!moment || !videoElement.value) return
  activeMoment.value = index
  seekTo(moment.start)
  stopAt.value = moment.end
  void videoElement.value.play().catch(() => {
    stopAt.value = null
  })
}

function resetResult() {
  if (isBusy.value) return
  result.value = null
  activeMoment.value = null
  stopAt.value = null
  status.value = videoUrl.value ? 'ready' : 'idle'
}

onBeforeUnmount(() => {
  if (videoUrl.value) URL.revokeObjectURL(videoUrl.value)
})

onMounted(async () => {
  try {
    const health = await getBackendHealth()
    backendStatus.value = health.status === 'ready' ? 'online' : 'offline'
    backendDevice.value = health.device
    backendModelLabel.value = health.model_label || 'CLIP'
    backendMaxDuration.value = health.max_video_duration ?? null
    backendMaxUploadMb.value = health.max_upload_mb ?? null
  } catch {
    backendStatus.value = 'offline'
  }
})
</script>

<template>
  <div class="app-shell">
    <header class="topbar">
      <div class="brand">
        <span class="brand-mark"><Video :size="19" stroke-width="2.4" /></span>
        <span>FrameSeek</span>
        <span class="brand-divider"></span>
        <span class="brand-subtitle">QD-DETR 视频语义定位</span>
      </div>
      <div class="system-state">
        <span class="status-dot" :class="{ offline: backendStatus === 'offline' }"></span>
        {{ backendLabel }}
        <span class="model-tag">{{ backendModelLabel }} · 2s</span>
      </div>
    </header>

    <main class="workspace">
      <section class="viewer-column">
        <div class="section-title-row">
          <div>
            <span class="eyebrow">VIDEO SOURCE</span>
            <h1>视频检索工作台</h1>
          </div>
          <button v-if="videoUrl" class="secondary-button" :disabled="isBusy" @click="openFilePicker">
            <Upload :size="16" /> 更换视频
          </button>
        </div>

        <div class="video-stage" :class="{ empty: !videoUrl }">
          <video
            v-if="videoUrl"
            ref="videoElement"
            :src="videoUrl"
            controls
            playsinline
            @loadedmetadata="onMetadataLoaded"
            @timeupdate="onTimeUpdate"
          ></video>
          <div v-else class="upload-state">
            <span class="upload-icon"><FileVideo2 :size="30" /></span>
            <h2>载入一段视频开始分析</h2>
            <p>支持 MP4、WebM、MOV{{ backendMaxDuration ? `，最长 ${backendMaxDuration} 秒` : '' }}{{ backendMaxUploadMb ? `，最大 ${backendMaxUploadMb} MB` : '' }}，检索时会上传到模型服务</p>
            <button class="primary-button" @click="openFilePicker">
              <Upload :size="17" /> 选择视频
            </button>
          </div>
          <input
            ref="fileInput"
            class="visually-hidden"
            type="file"
            accept=".mp4,.webm,.mov,video/mp4,video/webm,video/quicktime"
            :disabled="isBusy"
            @change="selectFile"
          />
        </div>

        <div v-if="videoUrl" class="file-strip">
          <span class="file-icon"><FileVideo2 :size="18" /></span>
          <span class="file-info">
            <strong>{{ videoName }}</strong>
            <small>{{ fileSizeLabel }} · {{ Math.round(duration) }} 秒</small>
          </span>
          <span v-if="durationTooLong" class="file-ready"><AlertCircle :size="14" /> 超过当前服务 {{ backendMaxDuration }} 秒上限</span>
          <span v-else-if="fileTooLarge" class="file-ready"><AlertCircle :size="14" /> 超过当前服务 {{ backendMaxUploadMb }} MB 上限</span>
          <span v-else class="file-ready"><Check :size="14" /> 已就绪</span>
        </div>

        <VideoTimeline
          v-if="result"
          :duration="result.duration"
          :current-time="currentTime"
          :moments="result.moments"
          :saliency-scores="result.saliency_scores"
          :clip-duration="result.clip_duration"
          :active-moment="activeMoment"
          @seek="seekTo"
          @select-moment="playMoment"
        />

        <div v-else-if="videoUrl" class="timeline-placeholder">
          <Sparkles :size="20" />
          <span>完成一次语义检索后，这里将显示相关片段与高光分布。</span>
        </div>
      </section>

      <aside class="inspector">
        <section class="query-section">
          <span class="eyebrow">TEXT QUERY</span>
          <h2>描述你想找的画面</h2>
          <label class="query-field">
            <span class="visually-hidden">视频内容描述</span>
            <textarea
              v-model="query"
              rows="4"
              maxlength="160"
              :disabled="isBusy"
              placeholder="例如：A person cuts a pizza."
              @input="resetResult"
              @keydown.ctrl.enter.prevent="runSearch"
            ></textarea>
            <span class="character-count">{{ query.length }}/160</span>
          </label>

          <div class="query-options">
            <label for="top-k">返回结果</label>
            <div class="select-wrap">
              <select id="top-k" v-model="topK" :disabled="isBusy" @change="resetResult">
                <option :value="1">Top 1</option>
                <option :value="3">Top 3</option>
                <option :value="5">Top 5</option>
              </select>
              <ChevronDown :size="15" />
            </div>
          </div>

          <button class="search-button" :disabled="!canSearch" @click="runSearch">
            <LoaderCircle v-if="isBusy" class="spin" :size="18" />
            <Search v-else :size="18" />
            {{ status === 'uploading' ? '正在上传视频…' : status === 'searching' ? '正在分析视频…' : '开始语义检索' }}
          </button>
        </section>

        <section class="result-section">
          <div class="results-heading">
            <div>
              <span class="eyebrow">MATCHES</span>
              <h2>检索结果</h2>
            </div>
            <div v-if="result" class="result-summary">
              <span class="result-count">{{ result.moments.length }} 个片段</span>
              <small>
                {{ (result.inference_time_ms / 1000).toFixed(2) }} 秒 ·
                {{ result.video_feature_cache_hit ? '已复用视频特征' : '已缓存视频特征' }}
              </small>
            </div>
          </div>

          <div v-if="status === 'idle' || status === 'ready'" class="inspector-empty">
            <span><Play :size="22" /></span>
            <p>{{ videoUrl ? '输入描述并开始检索' : '请先在左侧选择视频' }}</p>
          </div>

          <div v-else-if="status === 'uploading' || status === 'searching'" class="searching-state">
            <span class="scan-icon"><LoaderCircle class="spin" :size="24" /></span>
            <strong>{{ status === 'uploading' ? '正在上传视频' : '正在建立语义匹配' }}</strong>
            <p>{{ status === 'uploading' ? '上传完成后将自动开始分析' : 'QD-DETR 正在提取特征并执行推理' }}</p>
            <div class="loading-lines"><i></i><i></i><i></i></div>
          </div>

          <div v-else-if="status === 'error'" class="error-state">
            <AlertCircle :size="24" />
            <strong>无法完成检索</strong>
            <p>{{ errorMessage }}</p>
          </div>

          <ResultList
            v-else-if="result?.moments.length"
            :moments="result.moments"
            :active-moment="activeMoment"
            @select="playMoment"
          />

          <div v-else class="inspector-empty">
            <Search :size="22" />
            <p>没有找到匹配片段，请尝试更具体的描述。</p>
          </div>
        </section>

        <footer class="inspector-footer">
          <span>真实模型接口 · 视频特征缓存</span>
          <span>QD-DETR · {{ backendModelLabel }}</span>
        </footer>
      </aside>
    </main>
  </div>
</template>
